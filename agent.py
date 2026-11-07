"""
Stock Pipeline v2 — an agentic equity intelligence pipeline.

For every holding, a Claude agent decides which tools to call
(price, news, earnings, market benchmark, its own past reports),
digs deeper when something looks unusual, and submits a structured
report with a priority level. An editor step then writes a short
"what matters today" brief, and the digest is emailed.

Run:   python agent.py            -> analyze + email
       python agent.py --dry-run  -> analyze + save preview.html (no email)
"""

import json
import os
import re
import shutil
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import finnhub
import yfinance as yf
from anthropic import Anthropic
from dotenv import load_dotenv

# ─── Config ──────────────────────────────────────────────────────────
load_dotenv()
ROOT = Path(__file__).parent
PORTFOLIO_FILE = ROOT / "portfolio.json"
HISTORY_FILE = ROOT / "history.json"
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
PROPOSALS_FILE = ROOT / "proposals.json"   # log of past proposals (agent memory)
PROPOSAL_DIR = ROOT / "proposal"           # output the workflow turns into a PR
MAX_STEPS = 8          # max tool-call rounds per holding
MAX_REVIEW_STEPS = 12  # max tool-call rounds for the portfolio review
MAX_PROPOSALS = 2      # max add/remove proposals per run
PROPOSAL_COOLDOWN_DAYS = 14  # don't re-propose the same ticker within this window
HISTORY_KEEP = 30      # reports kept per holding
DRY_RUN = "--dry-run" in sys.argv
TODAY = datetime.now(timezone.utc).date()

fh = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
claude = Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# ─── Tools the agent can use ─────────────────────────────────────────
def is_crypto(ticker):
    return ticker.upper().endswith("-USD")


def get_price(ticker):
    """Latest price plus 1-day, 5-day and 1-month change."""
    closes = yf.Ticker(ticker).history(period="1mo")["Close"].dropna()
    if len(closes) < 2:
        return {"error": f"no price data for {ticker}"}
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    five = float(closes.iloc[-6]) if len(closes) >= 6 else float(closes.iloc[0])
    month = float(closes.iloc[0])
    pct = lambda a, b: round((a / b - 1) * 100, 2)
    return {
        "ticker": ticker,
        "price": round(last, 2),
        "change_1d_pct": pct(last, prev),
        "change_5d_pct": pct(last, five),
        "change_1m_pct": pct(last, month),
    }


def get_news(ticker, days=2):
    """Recent headlines. Crypto uses Finnhub's crypto feed filtered by coin."""
    days = max(1, min(int(days), 14))
    if is_crypto(ticker):
        coin = ticker.split("-")[0].lower()
        words = {"eth": ["ethereum", "eth", "ether"], "btc": ["bitcoin", "btc"]}.get(coin, [coin])
        pattern = re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)
        items = [
            n for n in fh.general_news("crypto")
            if pattern.search(f"{n.get('headline', '')} {n.get('summary', '')}")
        ]
    else:
        start = TODAY - timedelta(days=days)
        items = fh.company_news(ticker, _from=start.isoformat(), to=TODAY.isoformat())
    return [
        {
            "headline": n.get("headline"),
            "summary": (n.get("summary") or "")[:300],
            "source": n.get("source"),
            "date": datetime.fromtimestamp(n.get("datetime", 0), timezone.utc).strftime("%Y-%m-%d"),
        }
        for n in items[:8]
    ] or [{"note": f"no news found in the last {days} days"}]


def get_earnings(ticker):
    """Next earnings date within 45 days, if any."""
    if is_crypto(ticker):
        return {"note": "earnings do not apply to crypto"}
    cal = fh.earnings_calendar(
        _from=TODAY.isoformat(),
        to=(TODAY + timedelta(days=45)).isoformat(),
        symbol=ticker,
        international=False,
    )
    rows = cal.get("earningsCalendar", [])
    return rows[0] if rows else {"note": "no earnings scheduled in the next 45 days"}


def get_history(ticker, history):
    """The agent's own previous reports for this holding (its memory)."""
    return history.get(ticker, [])[-5:] or {"note": "no previous reports yet"}


TOOLS = [
    {
        "name": "get_price",
        "description": "Latest price and 1d/5d/1m % change. Works for stocks, ETFs (e.g. SPY, QQQ as market benchmarks) and crypto (e.g. ETH-USD).",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news",
        "description": "Recent news headlines for a ticker. Increase 'days' (max 14) when a price move has no explanation in recent news.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": "integer", "default": 2},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_earnings",
        "description": "Next scheduled earnings date (next 45 days) with EPS/revenue estimates.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_history",
        "description": "Your own previous reports on this ticker (last 5). Use it to detect sentiment shifts.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "submit_report",
        "description": "Submit the final report. Call exactly once, when your analysis is complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-3 sentences: what is happening and why."},
                "sentiment": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral", "Mixed"]},
                "sentiment_change": {"type": "string", "description": "e.g. 'Neutral → Bearish', or 'unchanged', or 'first report'."},
                "trend": {"type": "string", "description": "1 sentence: short-term trend or key catalyst to watch."},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "alert_reason": {"type": "string", "description": "Why this needs attention today. Empty string if priority is low."},
            },
            "required": ["summary", "sentiment", "sentiment_change", "trend", "priority", "alert_reason"],
        },
    },
]

SYSTEM_PROMPT = """You are an equity research agent monitoring one holding in a personal portfolio.

How to work:
- Start with get_price and get_history.
- Compare the move against the market (get_price on SPY or QQQ) to tell stock-specific moves from market-wide ones.
- Dig deeper only when it is warranted: a large move with no explanation -> get_news with more days;
  earnings may be close -> get_earnings.
- Base every claim on tool results. Never invent numbers or news.

Priority rules:
- high: stock-specific move larger than ~5% in a day, a sentiment flip vs. your last report, or earnings within 7 days.
- medium: notable news or a meaningful multi-day trend.
- low: nothing material.

Finish by calling submit_report exactly once."""


def run_tool(name, args, history):
    ticker = args.get("ticker", "")
    if name == "get_price":
        return get_price(ticker)
    if name == "get_news":
        return get_news(ticker, args.get("days", 2))
    if name == "get_earnings":
        return get_earnings(ticker)
    if name == "get_history":
        return get_history(ticker, history)
    return {"error": f"unknown tool {name}"}


# ─── The agent loop ──────────────────────────────────────────────────
REPORT_SCHEMA = next(t for t in TOOLS if t["name"] == "submit_report")["input_schema"]


def validate_report(report):
    """Return a list of problems with a submitted report (empty if valid)."""
    problems = []
    for field in REPORT_SCHEMA["required"]:
        value = report.get(field)
        if not isinstance(value, str):
            problems.append(f"missing or non-string field '{field}'")
            continue
        allowed = REPORT_SCHEMA["properties"][field].get("enum")
        if allowed and value not in allowed:
            problems.append(f"'{field}' must be one of {allowed}")
    return problems


def agent_loop(system, prompt, tools, finish_tool, validate, call_tool, max_steps, label):
    """Generic tool-use loop: the model calls tools until it submits `finish_tool`.

    Returns (submitted_input, steps). A submission that fails `validate` is sent
    back to the model with the problems so it can fix and resubmit.
    """
    messages = [{"role": "user", "content": prompt}]
    steps = []

    for _ in range(max_steps):
        response = claude.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if b.type == "tool_use"]

        if not calls:
            messages.append({"role": "user", "content": f"Please call {finish_tool} now."})
            continue

        results = []
        for call in calls:
            if call.name == finish_tool:
                submitted = dict(call.input)
                problems = validate(submitted)
                if not problems:
                    return submitted, steps
                # Bad submission: tell the agent what's wrong and let it resubmit.
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": "Rejected: " + "; ".join(problems) + f". Fix this and call {finish_tool} again.",
                    "is_error": True,
                })
                continue
            steps.append(f"{call.name}({call.input.get('ticker', '')})")
            try:
                output = call_tool(call.name, call.input)
            except Exception as e:  # tool failures go back to the agent, not crash the run
                output = {"error": str(e)}
            results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(output, default=str),
            })
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"{label}: agent did not finish within {max_steps} steps")


def run_agent(ticker, history):
    report, steps = agent_loop(
        system=SYSTEM_PROMPT,
        prompt=f"Analyze {ticker} for today's digest ({TODAY}).",
        tools=TOOLS,
        finish_tool="submit_report",
        validate=validate_report,
        call_tool=lambda name, args: run_tool(name, args, history),
        max_steps=MAX_STEPS,
        label=ticker,
    )
    report.update(ticker=ticker, steps=steps)
    return report


# ─── Portfolio review: propose adding/removing holdings ──────────────
# The agent only PROPOSES. The workflow turns a proposal into a pull request
# that edits portfolio.json; merging it is the human approval.
def get_peers(ticker):
    """Companies in the same industry (Finnhub peers)."""
    if is_crypto(ticker):
        return {"note": "peers are not available for crypto"}
    peers = [p for p in fh.company_peers(ticker) if p and p.upper() != ticker.upper()]
    return peers[:10] or {"note": f"no peers found for {ticker}"}


REVIEW_TOOLS = [
    next(t for t in TOOLS if t["name"] == n) for n in ("get_price", "get_news", "get_earnings", "get_history")
] + [
    {
        "name": "get_peers",
        "description": "Same-industry peer companies for a stock ticker. Use to find candidates to add.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "submit_proposals",
        "description": "Submit your proposals. Call exactly once. An empty list is the normal, expected result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "maxItems": MAX_PROPOSALS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "remove"]},
                            "ticker": {"type": "string"},
                            "reason": {"type": "string", "description": "2-3 sentences, citing tool results."},
                        },
                        "required": ["action", "ticker", "reason"],
                    },
                },
            },
            "required": ["proposals"],
        },
    },
]

REVIEW_PROMPT = f"""You review a personal watch-list portfolio and may propose changes to the list of holdings.
A human approves or rejects every proposal, so be conservative: most days you should propose NOTHING.

You get today's per-holding reports and your past proposals.
- Propose "remove" only for a current holding whose own reports show a sustained, evidence-backed problem
  (e.g. repeated Bearish reports with a clear negative catalyst), not for one bad day.
- Propose "add" only for a ticker you actually researched with tools (get_peers to find candidates,
  then get_price and get_news on the candidate), with a concrete reason it deserves watching.
- Never re-propose a ticker you proposed in the last {PROPOSAL_COOLDOWN_DAYS} days.
- At most {MAX_PROPOSALS} proposals. Base every claim on tool results. Never invent numbers or news.

Finish by calling submit_proposals exactly once (an empty list is fine)."""


def recent_proposal_tickers(log):
    cutoff = TODAY - timedelta(days=PROPOSAL_COOLDOWN_DAYS)
    return {p["ticker"].upper() for p in log if datetime.fromisoformat(p["date"]).date() >= cutoff}


def make_proposal_validator(holdings, log):
    held = {h.upper() for h in holdings}
    recent = recent_proposal_tickers(log)

    def validate(submitted):
        proposals = submitted.get("proposals")
        if not isinstance(proposals, list):
            return ["'proposals' must be a list"]
        if len(proposals) > MAX_PROPOSALS:
            return [f"at most {MAX_PROPOSALS} proposals"]
        problems = []
        for p in proposals:
            if not isinstance(p, dict) or not all(isinstance(p.get(k), str) and p.get(k) for k in ("action", "ticker", "reason")):
                problems.append("each proposal needs non-empty 'action', 'ticker' and 'reason'")
                continue
            t = p["ticker"].strip().upper()
            if p["action"] not in ("add", "remove"):
                problems.append(f"{t}: action must be 'add' or 'remove'")
            elif p["action"] == "remove" and t not in held:
                problems.append(f"{t}: cannot remove, it is not a current holding")
            elif p["action"] == "add" and t in held:
                problems.append(f"{t}: cannot add, it is already a holding")
            elif p["action"] == "add" and "error" in get_price(t):
                problems.append(f"{t}: no price data, not a valid ticker")
            if t in recent:
                problems.append(f"{t}: already proposed in the last {PROPOSAL_COOLDOWN_DAYS} days")
        return problems

    return validate


def review_portfolio(holdings, reports, history, log):
    slim = [{k: r[k] for k in ("ticker", "sentiment", "sentiment_change", "priority", "summary")} for r in reports]
    prompt = (
        f"Today is {TODAY}. Current holdings: {holdings}.\n\n"
        f"Today's reports:\n{json.dumps(slim)}\n\n"
        f"Your past proposals (most recent last):\n{json.dumps(log[-20:])}"
    )

    def call_tool(name, args):
        if name == "get_peers":
            return get_peers(args.get("ticker", ""))
        return run_tool(name, args, history)

    submitted, steps = agent_loop(
        system=REVIEW_PROMPT,
        prompt=prompt,
        tools=REVIEW_TOOLS,
        finish_tool="submit_proposals",
        validate=make_proposal_validator(holdings, log),
        call_tool=call_tool,
        max_steps=MAX_REVIEW_STEPS,
        label="portfolio review",
    )
    proposals = [
        {"action": p["action"], "ticker": p["ticker"].strip().upper(), "reason": p["reason"].strip()}
        for p in submitted["proposals"]
    ]
    return proposals, steps


def write_proposal_files(holdings, proposals):
    """Write proposal/ for the workflow to turn into a pull request."""
    new = [h for h in holdings if h.upper() not in {p["ticker"] for p in proposals if p["action"] == "remove"}]
    new += [p["ticker"] for p in proposals if p["action"] == "add"]
    PROPOSAL_DIR.mkdir(exist_ok=True)
    (PROPOSAL_DIR / "portfolio.json").write_text(json.dumps({"holdings": new}, indent=2) + "\n")
    summary = ", ".join(f"{p['action']} {p['ticker']}" for p in proposals)
    (PROPOSAL_DIR / "title.txt").write_text(f"Portfolio proposal {TODAY}: {summary}")
    body = [
        "The portfolio-review agent proposes these changes to `portfolio.json`.",
        "",
        "**Merge this PR to approve. Close it to reject.**",
        "",
    ]
    for p in proposals:
        body += [f"### {p['action'].capitalize()} {p['ticker']}", p["reason"], ""]
    body += [f"Holdings after this change: {', '.join(new)}", "", "_Generated by Stock Pipeline. Not investment advice._"]
    (PROPOSAL_DIR / "body.md").write_text("\n".join(body) + "\n")


# ─── Editor step: one brief across the whole portfolio ───────────────
def write_brief(reports):
    slim = [
        {k: r[k] for k in ("ticker", "sentiment", "sentiment_change", "priority", "alert_reason", "summary")}
        for r in reports
    ]
    response = claude.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": "You are the editor of a daily portfolio digest. From these per-holding reports, "
                       "write 2-3 short bullet points (plain text, each starting with '- ') on what matters "
                       "most today across the portfolio. No intro, no outro.\n\n" + json.dumps(slim),
        }],
    )
    # The response may contain non-text blocks (e.g. thinking); keep only text.
    text = "\n".join(b.text for b in response.content if b.type == "text")
    bullets = (line.lstrip("-• ").strip() for line in text.splitlines())
    return [b for b in bullets if b]


# ─── Memory ──────────────────────────────────────────────────────────
def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_history(history, reports, prices):
    for r in reports:
        entry = {
            "date": TODAY.isoformat(),
            "price": prices.get(r["ticker"]),
            "sentiment": r["sentiment"],
            "priority": r["priority"],
            "summary": r["summary"],
        }
        past = [e for e in history.get(r["ticker"], []) if e["date"] != entry["date"]]
        history[r["ticker"]] = (past + [entry])[-HISTORY_KEEP:]
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ─── Email ───────────────────────────────────────────────────────────
COLORS = {"Bullish": "#16a34a", "Bearish": "#dc2626", "Mixed": "#ca8a04", "Neutral": "#6b7280"}
PRIORITY_BADGE = {"high": ("HIGH", "#dc2626"), "medium": ("MEDIUM", "#ca8a04"), "low": ("LOW", "#9ca3af")}


def build_proposals_html(proposals):
    if not proposals:
        return ""
    repo = os.getenv("GITHUB_REPOSITORY")
    where = (f'<a href="https://github.com/{repo}/pulls">the pull request on GitHub</a>' if repo
             else "the pull request on GitHub")
    items = "".join(
        f'<li style="margin:6px 0"><strong>{p["action"].capitalize()} {p["ticker"]}</strong> — {p["reason"]}</li>'
        for p in proposals
    )
    return f"""<div style="border:2px solid #2563eb;border-radius:8px;padding:12px 16px;margin:16px 0">
<strong>🧭 Proposed portfolio changes</strong><ul style="margin:8px 0;padding-left:20px">{items}</ul>
<p style="margin:4px 0;font-size:13px;color:#374151">Approve by merging {where}; close it to reject.</p></div>"""


def build_html(reports, brief, prices, proposals=()):
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    html = f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;max-width:700px;margin:auto;padding:16px;color:#111">
<h2 style="border-bottom:2px solid #111;padding-bottom:8px">📊 Stock Pipeline — {date_str}</h2>
<div style="background:#f3f4f6;border-radius:8px;padding:12px 16px;margin:16px 0">
<strong>What matters today</strong><ul style="margin:8px 0;padding-left:20px">
{''.join(f'<li style="margin:4px 0">{b}</li>' for b in brief)}
</ul></div>{build_proposals_html(proposals)}"""
    for r in reports:
        p = prices.get(r["ticker"]) or {}
        chg = p.get("change_1d_pct", 0)
        label, badge_color = PRIORITY_BADGE[r["priority"]]
        alert = f'<p style="margin:8px 0;color:#dc2626"><strong>⚠ Alert:</strong> {r["alert_reason"]}</p>' if r["alert_reason"] else ""
        html += f"""
<div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0">
  <div>
    <span style="font-size:18px;font-weight:700">{r['ticker']}</span>
    <span style="font-size:16px;font-weight:600;margin-left:10px">${p.get('price', '—')}</span>
    <span style="color:{'#16a34a' if chg >= 0 else '#dc2626'};margin-left:6px">{chg:+.2f}%</span>
    <span style="float:right;background:{badge_color};color:#fff;font-size:11px;font-weight:700;padding:3px 8px;border-radius:4px">{label}</span>
  </div>
  {alert}
  <p style="margin:10px 0"><strong>Summary:</strong> {r['summary']}</p>
  <p style="margin:6px 0"><strong>Sentiment:</strong>
     <span style="color:{COLORS.get(r['sentiment'], '#6b7280')};font-weight:600">{r['sentiment']}</span>
     <span style="color:#6b7280"> ({r['sentiment_change']})</span></p>
  <p style="margin:6px 0"><strong>Trend:</strong> {r['trend']}</p>
  <p style="margin:6px 0;color:#9ca3af;font-size:11px">Agent steps: {' → '.join(r['steps']) or '—'}</p>
</div>"""
    html += "<p style='color:#6b7280;font-size:12px'>Generated by Stock Pipeline (agentic v2). Not investment advice.</p></body></html>"
    return html


def send_email(html, alerts):
    msg = MIMEMultipart("alternative")
    tag = f" · ⚠ {alerts} alert{'s' if alerts != 1 else ''}" if alerts else ""
    msg["Subject"] = f"📊 Stock Pipeline — {datetime.now().strftime('%b %d')}{tag}"
    msg["From"] = os.environ["GMAIL_ADDRESS"]
    msg["To"] = os.environ["RECIPIENT_EMAIL"]
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"])
        server.send_message(msg)


# ─── Main ────────────────────────────────────────────────────────────
def main():
    holdings = load_json(PORTFOLIO_FILE, {"holdings": []})["holdings"]
    history = load_json(HISTORY_FILE, {})
    print(f"\n🚀 Stock Pipeline v2 — {TODAY} — {len(holdings)} holdings, model {MODEL}\n")

    reports, prices = [], {}
    for ticker in holdings:
        print(f"🔎 {ticker}")
        try:
            prices[ticker] = get_price(ticker)
            report = run_agent(ticker, history)
            reports.append(report)
            print(f"   {' → '.join(report['steps'])}")
            print(f"   ✓ {report['sentiment']} · priority {report['priority']}")
        except Exception as e:
            print(f"   ✗ {e}")

    if not reports:
        print("\n⚠ No reports generated, nothing sent.")
        sys.exit(1)

    order = {"high": 0, "medium": 1, "low": 2}
    reports.sort(key=lambda r: order[r["priority"]])
    brief = write_brief(reports)

    # Portfolio review: may propose adding/removing holdings (human approves via PR).
    shutil.rmtree(PROPOSAL_DIR, ignore_errors=True)
    proposal_log = load_json(PROPOSALS_FILE, [])
    proposals = []
    if os.getenv("OPEN_PROPOSAL", "0") not in ("", "0"):
        print("\n🧭 Portfolio review skipped: a proposal pull request is still open.")
    else:
        print("\n🧭 Portfolio review")
        try:
            proposals, steps = review_portfolio(holdings, reports, history, proposal_log)
            print(f"   {' → '.join(steps) or 'no tool calls'}")
            for p in proposals:
                print(f"   ➜ propose {p['action']} {p['ticker']}")
            if not proposals:
                print("   ✓ no changes proposed")
        except Exception as e:  # a failed review must not block the daily digest
            print(f"   ✗ review failed: {e}")
            proposals = []
    if proposals:
        write_proposal_files(holdings, proposals)
        if not DRY_RUN:
            proposal_log += [{"date": TODAY.isoformat(), **p} for p in proposals]
            PROPOSALS_FILE.write_text(json.dumps(proposal_log[-100:], indent=2) + "\n")

    html = build_html(reports, brief, prices, proposals)
    alerts = sum(r["priority"] == "high" for r in reports)

    if DRY_RUN:
        (ROOT / "preview.html").write_text(html)
        print("\n👀 Dry run: open preview.html to see the email.")
    else:
        send_email(html, alerts)
        print(f"\n✅ Email sent — {len(reports)} holdings, {alerts} alerts")

    save_history(history, reports, prices)


if __name__ == "__main__":
    main()
