"""
Stock Pipeline — Daily portfolio digest
Fetches news + price data, asks Claude to analyze each stock,
and emails you a clean morning briefing.
"""

import os
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yfinance as yf
import finnhub
from anthropic import Anthropic
from dotenv import load_dotenv

# ─── Load secrets from .env ──────────────────────────────────────────
load_dotenv()
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GMAIL = os.getenv("GMAIL_ADDRESS")
GMAIL_PW = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT = os.getenv("RECIPIENT_EMAIL")
TICKERS = os.getenv("TICKERS").split(",")

# ─── Initialize API clients ──────────────────────────────────────────
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
claude = Anthropic(api_key=ANTHROPIC_KEY)


def get_price_data(ticker):
    """Get yesterday's close, today's price, and % change."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if len(hist) < 2:
        return None
    prev_close = hist["Close"].iloc[-2]
    current = hist["Close"].iloc[-1]
    pct_change = ((current - prev_close) / prev_close) * 100
    return {
        "current": round(float(current), 2),
        "prev_close": round(float(prev_close), 2),
        "pct_change": round(float(pct_change), 2),
    }


def get_news(ticker, days_back=2):
    """Fetch recent news headlines for a ticker."""
    today = datetime.now().date()
    start = today - timedelta(days=days_back)
    try:
        news = finnhub_client.company_news(
            ticker, _from=start.isoformat(), to=today.isoformat()
        )
        return news[:5] if news else []
    except Exception as e:
        print(f"  ⚠ News fetch failed for {ticker}: {e}")
        return []


def analyze_with_claude(ticker, price_data, news):
    """Use Claude to generate summary + sentiment + trend."""
    if not news:
        return {
            "summary": "No significant news in the last 48 hours.",
            "sentiment": "Neutral",
            "trend": "No fresh catalyst — monitor for next earnings or sector news.",
        }

    headlines = "\n".join(
        [f"- {n['headline']}: {n.get('summary', '')[:200]}" for n in news]
    )

    prompt = f"""You are a concise equity analyst. Analyze {ticker}.

Price action: ${price_data['current']} ({price_data['pct_change']:+.2f}% vs prev close)

Recent news headlines:
{headlines}

Respond in EXACTLY this format (no extra text, no markdown):

SUMMARY: [2-3 sentences on what is happening with this stock]
SENTIMENT: [Bullish / Bearish / Neutral / Mixed]
TREND: [1 sentence on the short-term trend or key catalyst to watch]
"""

    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text
    result = {"summary": "", "sentiment": "", "trend": ""}
    for line in text.split("\n"):
        if line.startswith("SUMMARY:"):
            result["summary"] = line.replace("SUMMARY:", "").strip()
        elif line.startswith("SENTIMENT:"):
            result["sentiment"] = line.replace("SENTIMENT:", "").strip()
        elif line.startswith("TREND:"):
            result["trend"] = line.replace("TREND:", "").strip()
    return result


def sentiment_color(sentiment):
    s = sentiment.lower()
    if "bullish" in s:
        return "#16a34a"
    if "bearish" in s:
        return "#dc2626"
    if "mixed" in s:
        return "#ca8a04"
    return "#6b7280"


def build_email_html(reports):
    """Build a clean HTML email."""
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    html = f"""
    <html><body style="font-family: -apple-system, sans-serif; max-width: 700px; margin: auto; padding: 16px;">
    <h2 style="border-bottom: 2px solid #111; padding-bottom: 8px;">
    📊 Stock Pipeline — {today_str}
    </h2>
    """
    for r in reports:
        change_color = "#16a34a" if r["price"]["pct_change"] >= 0 else "#dc2626"
        html += f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0;">
          <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <h3 style="margin: 0;">{r['ticker']}</h3>
            <span style="font-size: 18px; font-weight: 600;">
              ${r['price']['current']}
              <span style="color: {change_color}; font-size: 14px; margin-left: 8px;">
                {r['price']['pct_change']:+.2f}%
              </span>
            </span>
          </div>
          <p style="margin: 12px 0;"><strong>Summary:</strong> {r['analysis']['summary']}</p>
          <p style="margin: 8px 0;">
            <strong>Sentiment:</strong>
            <span style="color: {sentiment_color(r['analysis']['sentiment'])}; font-weight: 600;">
              {r['analysis']['sentiment']}
            </span>
          </p>
          <p style="margin: 8px 0;"><strong>Trend:</strong> {r['analysis']['trend']}</p>
        </div>
        """
    html += "<p style='color: #6b7280; font-size: 12px; margin-top: 24px;'>Generated by your Stock Pipeline bot. Not investment advice.</p>"
    html += "</body></html>"
    return html


def send_email(html_content):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📊 Stock Pipeline — {datetime.now().strftime('%b %d')}"
    msg["From"] = GMAIL
    msg["To"] = RECIPIENT
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL, GMAIL_PW)
        server.send_message(msg)


def main():
    print(f"\n🚀 Stock Pipeline starting — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    reports = []
    for ticker in TICKERS:
        ticker = ticker.strip()
        print(f"Processing {ticker}...")
        try:
            price = get_price_data(ticker)
            if not price:
                print(f"  ⚠ No price data for {ticker}, skipping")
                continue
            news = get_news(ticker)
            analysis = analyze_with_claude(ticker, price, news)
            reports.append({"ticker": ticker, "price": price, "analysis": analysis})
            print(f"  ✓ {ticker}: ${price['current']} ({price['pct_change']:+.2f}%) → {analysis['sentiment']}")
        except Exception as e:
            print(f"  ✗ Error on {ticker}: {e}")

    if reports:
        html = build_email_html(reports)
        send_email(html)
        print(f"\n✅ Email sent with {len(reports)} stocks to {RECIPIENT}")
    else:
        print("\n⚠ No reports generated, email not sent")


if __name__ == "__main__":
    main()