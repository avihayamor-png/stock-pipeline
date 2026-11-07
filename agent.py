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
MAX_STEPS = 8          # max tool-call rounds per holding
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


def run_agent(ticker, history):
    messages = [{"role": "user", "content": f"Analyze {ticker} for today's digest ({TODAY})."}]
    steps = []

    for _ in range(MAX_STEPS):
        response = claude.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if b.type == "tool_use"]

        if not calls:
            messages.append({"role": "user", "content": "Please call submit_report now."})
            continue

        results = []
        for call in calls:
            if call.name == "submit_report":
                report = dict(call.input)
                problems = validate_report(report)
                if not problems:
                    report.update(ticker=ticker, steps=steps)
                    return report
                # Bad report: tell the agent what's wrong and let it resubmit.
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": "Report rejected: " + "; ".join(problems) + ". Call submit_report again with all fields.",
                    "is_error": True,
                })
                continue
            steps.append(f"{call.name}({call.input.get('ticker', '')})")
            try:
                output = run_tool(call.name, call.input, history)
            except Exception as e:  # tool failures go back to the agent, not crash the run
                output = {"error": str(e)}
            results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(output, default=str),
            })
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"{ticker}: agent did not finish within {MAX_STEPS} steps")


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


def build_html(reports, brief, prices):
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    html = f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;max-width:700px;margin:auto;padding:16px;color:#111">
<h2 style="border-bottom:2px solid #111;padding-bottom:8px">📊 Stock Pipeline — {date_str}</h2>
<div style="background:#f3f4f6;border-radius:8px;padding:12px 16px;margin:16px 0">
<strong>What matters today</strong><ul style="margin:8px 0;padding-left:20px">
{''.join(f'<li style="margin:4px 0">{b}</li>' for b in brief)}
</ul></div>"""
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
    html = build_html(reports, brief, prices)
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
