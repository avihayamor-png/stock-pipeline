# 📊 Stock Pipeline — an agentic equity intelligence agent

An AI agent that monitors my personal portfolio every weekday and emails me a prioritized research digest.
It runs on GitHub Actions — no server, no laptop needed.

## What makes it an agent (not a script)

| v1 — scripted pipeline | v2 — agent |
|---|---|
| Fixed steps: fetch price → fetch news → summarize | Claude **decides which tools to call** per holding |
| Same depth for every stock | **Digs deeper** when a move is unexplained (more news days, earnings check) |
| No context between days | **Memory**: compares today vs. its own past reports (`history.json`) |
| Every stock looks equally important | **Prioritizes**: high / medium / low, with an alert reason |
| One summary per stock | **Editor step** writes a cross-portfolio "What matters today" brief |
| Laptop cron (fails when the Mac is off) | **GitHub Actions** schedule, runs in the cloud |

## How it works

```
GitHub Actions (Mon–Fri, 14:00 Israel)
        │
        ▼
for each holding in portfolio.json:
   Claude agent loop (max 8 steps)
     ├─ get_price(ticker | SPY | QQQ)   ← yfinance
     ├─ get_news(ticker, days)          ← Finnhub
     ├─ get_earnings(ticker)            ← Finnhub
     ├─ get_history(ticker)             ← its own memory
     └─ submit_report → sentiment, sentiment change, trend, priority, alert
        │
        ▼
Editor step → "What matters today" brief
        │
        ▼
HTML email (Gmail SMTP) + history.json committed back (memory)
```

## Tools the agent can call

- **get_price**: price + 1d/5d/1m change. Also used on SPY/QQQ to separate stock-specific moves from market moves.
- **get_news**: company news, or the crypto feed for coins like ETH-USD.
- **get_earnings**: next earnings date and estimates.
- **get_history**: the agent's last 5 reports on that ticker, to detect sentiment shifts.
- **submit_report**: structured output (JSON schema), so the email never breaks on free text.

## Portfolio proposals (human-in-the-loop)

After the daily reports, a second **portfolio-review agent** looks across all holdings and may propose
adding or removing tickers. It can call `get_peers` (same-industry companies, via Finnhub) plus the tools above.

- It only **proposes**. The workflow turns a proposal into a **pull request** that edits `portfolio.json`.
  **Merge the PR to approve, close it to reject.** The email also shows the proposal.
- Guardrails: at most 2 proposals per run, an empty proposal is the normal result, every claim must come
  from tool results, and code validates each proposal (real ticker, not already held, no repeat within 14 days).
- Only one proposal PR is open at a time; the review is skipped while one is pending.
- Past proposals are kept in `proposals.json`, so the agent remembers what it already suggested.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
python agent.py --dry-run   # writes preview.html, no email
python agent.py             # sends the email
```

Edit `portfolio.json` to change the holdings.

## Run it on GitHub (scheduled)

1. Repo → **Settings → Secrets and variables → Actions** → add:
   `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL`
2. **Actions → Daily digest → Run workflow** to test.
3. After that it runs Mon–Fri at 12:00 UTC.

## Stack

Python · Anthropic Claude API (tool use) · Finnhub · yfinance · GitHub Actions · Gmail SMTP

---
*Personal project. Not investment advice.*
