import requests
from bs4 import BeautifulSoup
import os
from flask import Flask, jsonify
from flask_cors import CORS
import datetime
from collections import deque
import re
import logging
import tweepy
import resend
import openai
from datetime import datetime as dt

# ---------- AI PROVIDERS ----------
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2")

groq_client = None
if GROQ_API_KEY and GROQ_API_KEY.strip():
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logging.info("Groq client initialized with primary key")
    except Exception as e:
        logging.warning(f"Failed to initialize Groq with primary key: {e}")
elif GROQ_API_KEY_2 and GROQ_API_KEY_2.strip():
    try:
        groq_client = Groq(api_key=GROQ_API_KEY_2)
        logging.info("Groq client initialized with secondary key")
    except Exception as e:
        logging.warning(f"Failed to initialize Groq with secondary key: {e}")
else:
    logging.warning("No valid GROQ_API_KEY found; Groq will be skipped.")

from google import genai
GEMINI_KEY_1 = os.environ.get("GEMINI_API_KEY_1")
GEMINI_KEY_2 = os.environ.get("GEMINI_API_KEY_2")
GEMINI_KEY_3 = os.environ.get("GEMINI_API_KEY_3")

# ---------- ADDITIONAL AI PROVIDERS ----------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

app = Flask(__name__)
CORS(app, origins="*")
logging.basicConfig(level=logging.INFO)

score_history = deque(maxlen=7)

# ---------- HELPERS ----------
def fetch_soup(url, timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    return BeautifulSoup(r.text, 'html.parser')

def extract_text(soup, max_chars=4000):
    for selector in ['article', 'div#content', 'div.content', 'main', 'div.article-body', 'div.story-body']:
        tag = soup.select_one(selector)
        if tag:
            text = tag.get_text(separator=' ', strip=True)
            if len(text) > 100:
                return text[:max_chars]
    paragraphs = soup.find_all('p')
    if paragraphs:
        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        if len(text) > 100:
            return text[:max_chars]
    body = soup.find('body')
    if body:
        return body.get_text(separator=' ', strip=True)[:max_chars]
    return ""

def looks_like_individual_doc(url):
    if any(kw in url for kw in ['rss', '.xml']):
        return False
    return True

# ---------- SOURCE SCRAPERS (INFLATION‑SPECIFIC) ----------
def scrape_earnings_calls():
    sources = []
    try:
        soup = fetch_soup("https://www.fool.com/earnings-call-transcripts/")
        items = soup.select('a[href*="/earnings-call-transcripts/"]')
        for a in items[:3]:
            href = a.get('href')
            if href:
                full_url = href if href.startswith('http') else "https://www.fool.com" + href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'earnings_call', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} earnings call transcripts")
    except Exception as e:
        logging.error(f"Earnings calls scrape error: {e}")
    return sources

def scrape_ism_report():
    sources = []
    try:
        soup = fetch_soup("https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/")
        items = soup.select('a[href*="ROB"]')
        for a in items[:1]:
            href = a.get('href')
            if href:
                full_url = href if href.startswith('http') else "https://www.ismworld.org" + href
                title = a.get_text(strip=True) or "ISM Manufacturing PMI"
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'ism_report', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} ISM report links")
    except Exception as e:
        logging.error(f"ISM scrape error: {e}")
    return sources

def scrape_bls_cpi():
    sources = []
    try:
        soup = fetch_soup("https://www.bls.gov/news.release/cpi.nr0.htm")
        sources.append({
            'type': 'bls_cpi',
            'title': 'BLS CPI News Release',
            'url': 'https://www.bls.gov/news.release/cpi.nr0.htm'
        })
        logging.info("BLS CPI release added")
    except Exception as e:
        logging.error(f"BLS CPI scrape error: {e}")
    return sources

def scrape_shipping_statements():
    sources = []
    try:
        soup = fetch_soup("https://www.maersk.com/news/articles")
        items = soup.select('a[href*="/news/articles/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://www.maersk.com" + href if href.startswith('/') else href
                title = a.get_text(strip=True) or "Maersk Update"
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'shipping', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} shipping statements")
    except Exception as e:
        logging.error(f"Shipping scrape error: {e}")
    return sources

def scrape_inflation_news():
    sources = []
    try:
        soup = fetch_soup("https://www.reuters.com/markets/commodities/")
        items = soup.select('a[href*="/markets/commodities/"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://www.reuters.com" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'inflation_news', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} inflation news headlines")
    except Exception as e:
        logging.error(f"Inflation news scrape error: {e}")
    return sources

def scrape_commodity_prices():
    sources = []
    try:
        soup = fetch_soup("https://finance.yahoo.com/news/")
        items = soup.select('a[href*="gold"]')
        for a in items[:2]:
            href = a.get('href')
            if href:
                full_url = "https://finance.yahoo.com" + href if href.startswith('/') else href
                title = a.get_text(strip=True)
                if not any(s['url'] == full_url for s in sources):
                    sources.append({'type': 'commodity', 'title': title, 'url': full_url})
        logging.info(f"Scraped {len(sources)} commodity price headlines")
    except Exception as e:
        logging.error(f"Commodity scrape error: {e}")
    return sources

# ---------- AI SCORING (INFLATION PROMPT) ----------
def score_text_with_ai(text):
    if not text:
        return None
    prompt = f"""
You are an inflation sentiment analyzer. Rate the following text on a scale from 0 (extremely deflationary / disinflationary, suggesting falling prices and weak demand) to 100 (extremely inflationary, suggesting accelerating price pressures and overheating). Consider language about input costs, wage growth, pricing power, supply chain bottlenecks, and demand strength. Return ONLY the number, no explanation.

Text:
{text[:3000]}
"""
    if groq_client:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=5
            )
            score_str = chat_completion.choices[0].message.content.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Groq): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Groq failed ({e}), falling back to Gemini-1...")
    if GEMINI_KEY_1:
        try:
            gemini_client = genai.Client(api_key=GEMINI_KEY_1)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 5}
            )
            score_str = response.text.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Gemini-1): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Gemini-1 failed ({e}), falling back to Gemini-2...")
    if GEMINI_KEY_2:
        try:
            gemini_client = genai.Client(api_key=GEMINI_KEY_2)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 5}
            )
            score_str = response.text.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Gemini-2): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Gemini-2 failed ({e}), falling back to Gemini-3...")
    if GEMINI_KEY_3:
        try:
            gemini_client = genai.Client(api_key=GEMINI_KEY_3)
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0, "max_output_tokens": 5}
            )
            score_str = response.text.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (Gemini-3): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"Gemini-3 failed ({e}), falling back to DeepSeek...")
    
    # Tier 5: DeepSeek
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.strip():
        try:
            deepseek_client = openai.OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1"
            )
            response = deepseek_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )
            score_str = response.choices[0].message.content.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (DeepSeek): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.warning(f"DeepSeek failed ({e}), falling back to OpenRouter...")
    
    # Tier 6: OpenRouter
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip():
        try:
            openrouter_client = openai.OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
            response = openrouter_client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )
            score_str = response.choices[0].message.content.strip()
            digits = re.findall(r'\d+', score_str)
            if digits:
                score = int(digits[0])
                logging.info(f"AI score (OpenRouter): {score}")
                return max(0, min(100, score))
        except Exception as e:
            logging.error(f"OpenRouter failed ({e})")
            return None

    # If we've made it here, all AI providers have failed
    logging.error("All AI providers failed (Groq, Gemini 1-3, DeepSeek, OpenRouter)")
    return None

# ---------- MARKET EXPECTATION (BREAKEVEN INFLATION + COMMODITIES) ----------
def compute_market_gip():
    be_rate = None
    gold_change = 0
    oil_change = 0

    fred_api_key = os.environ.get("FRED_API_KEY")
    if fred_api_key:
        try:
            resp = requests.get(
                f"https://api.stlouisfed.org/fred/series/observations?series_id=T5YIE&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1",
                timeout=15
            )
            if resp.status_code == 200:
                obs = resp.json().get("observations", [])
                if obs:
                    be_rate = float(obs[0].get("value", 0) or 0)
        except Exception as e:
            logging.warning(f"FRED breakeven error: {e}")

    try:
        gold_url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&range=1d"
        resp = requests.get(gold_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("chart", {}).get("result", [])
            if data:
                meta = data[0].get("meta", {})
                prev = meta.get("previousClose")
                curr = meta.get("regularMarketPrice")
                if prev and curr:
                    gold_change = ((curr / prev) - 1) * 100
    except Exception as e:
        logging.warning(f"Gold price error: {e}")

    try:
        oil_url = "https://query1.finance.yahoo.com/v8/finance/chart/CL=F?interval=1d&range=1d"
        resp = requests.get(oil_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("chart", {}).get("result", [])
            if data:
                meta = data[0].get("meta", {})
                prev = meta.get("previousClose")
                curr = meta.get("regularMarketPrice")
                if prev and curr:
                    oil_change = ((curr / prev) - 1) * 100
    except Exception as e:
        logging.warning(f"Oil price error: {e}")

    be_score = 50
    if be_rate is not None and be_rate > 0:
        be_score = min(100, max(0, (be_rate - 1.5) * (100 / 2.0)))

    gold_score = min(100, max(0, 50 + gold_change * 100))
    oil_score = min(100, max(0, 50 + oil_change * 100))

    market_gip = round((be_score + gold_score + oil_score) / 3, 1)

    if market_gip <= 20:
        label = "Extremely Deflationary"
    elif market_gip <= 40:
        label = "Deflationary"
    elif market_gip <= 60:
        label = "Neutral"
    elif market_gip <= 80:
        label = "Inflationary"
    else:
        label = "Extremely Inflationary"

    return market_gip, label, {
        "breakeven_rate": be_rate,
        "gold_change": gold_change,
        "oil_change": oil_change,
        "be_score": be_score,
        "gold_score": gold_score,
        "oil_score": oil_score
    }

# ---------- COMBINED PIPELINE ----------
def compute_daily_gip():
    all_sources = []
    all_sources.extend(scrape_earnings_calls())
    all_sources.extend(scrape_ism_report())
    all_sources.extend(scrape_bls_cpi())
    all_sources.extend(scrape_shipping_statements())
    all_sources.extend(scrape_inflation_news())
    all_sources.extend(scrape_commodity_prices())

    scores = []
    total_chars = 0
    sources_detail = []
    for src in all_sources:
        try:
            soup = fetch_soup(src['url'])
            text = extract_text(soup)
            if text:
                score = score_text_with_ai(text)
                if score is not None:
                    scores.append(score)
                    total_chars += len(text)
                    sources_detail.append({
                        'type': src['type'],
                        'title': src['title'],
                        'url': src['url'],
                        'chars': len(text),
                        'speaker': 'N/A'
                    })
                    logging.info(f"Scored {src['type']}: {score}")
        except Exception as e:
            logging.error(f"Error processing {src['url']}: {e}")

    if not scores:
        return None, None, []

    raw = sum(scores) / len(scores)
    if len(score_history) > 0:
        smoothed = round(sum(score_history) / len(score_history), 1)
    else:
        smoothed = round(raw, 1)
    score_history.append(raw)

    num_sources = len(sources_detail)
    if num_sources >= 4 and total_chars > 8000:
        confidence = "HIGH"
    elif num_sources >= 2 and total_chars > 3000:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return smoothed, confidence, sources_detail

# ---------- ROUTES ----------
@app.route('/health')
def health():
    return "OK"

@app.route('/ping')
def ping():
    score, confidence, sources = compute_daily_gip()
    if score is None:
        return jsonify({"status": "error", "message": "No data"}), 500
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({"status": "ok", "score": score, "timestamp": ts})

@app.route('/api/gip_latest')
def gip_latest():
    score, confidence, sources = compute_daily_gip()
    if score is None:
        return jsonify({"error": "No data available"}), 500
    prev = list(score_history)
    change = round(score - prev[-2], 1) if len(prev) > 1 else 0
    raw_score = prev[-1] if prev else score
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({
        "index": "G-Pulse (GIP)",
        "score": score,
        "raw_score": raw_score,
        "change": change,
        "confidence": confidence,
        "sources": sources,
        "timestamp": ts
    })

@app.route('/api/market_gip')
def market_gip():
    score, label, components = compute_market_gip()
    if score is None:
        return jsonify({"error": "Market data unavailable"}), 500
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({
        "index": "Market Expectation (GIP‑M)",
        "score": score,
        "label": label,
        "components": components,
        "timestamp": ts
    })

@app.route('/')
def home():
    return "G-Pulse (GIP) Global Inflation Pulse is live. Use /api/gip_latest"

@app.route('/post_tweet')
def auto_post():
    try:
        score, confidence, sources = compute_daily_gip()
        if score is None:
            return jsonify({"status": "No score available"})
        prev = list(score_history)
        change = round(score - prev[-2], 1) if len(prev) > 1 else 0
        if change == 0:
            arrow = "—"
        elif change > 0:
            arrow = f"▲{abs(change)}"
        else:
            arrow = f"▼{abs(change)}"
        if score <= 20:
            label = "Extremely Deflationary"
        elif score <= 40:
            label = "Deflationary"
        elif score <= 60:
            label = "Neutral"
        elif score <= 80:
            label = "Inflationary"
        else:
            label = "Extremely Inflationary"
        sources_count = len(sources) if sources else 0
        tweet_text = (
            f"🌍 GIP today: {score} {arrow} — {label}\n"
            f"Confidence: {confidence} | Sources: {sources_count}"
        )
        client = tweepy.Client(
            consumer_key=os.environ["X_CONSUMER_KEY"],
            consumer_secret=os.environ["X_CONSUMER_SECRET"],
            access_token=os.environ["X_ACCESS_TOKEN"],
            access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"]
        )
        response = client.create_tweet(text=tweet_text)
        logging.info(f"Tweet posted: {response.data['id']}")
        return jsonify({"status": "Tweet posted successfully"})
    except Exception as e:
        logging.error(f"Auto‑post failed: {e}")
        return jsonify({"status": f"Error posting tweet: {e}"})

@app.route('/api/gip_drivers')
def gip_drivers():
    """Return the core inflation drivers: CPI, Core PCE, 5‑year breakeven."""
    import requests
    import os
    
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        return jsonify({"error": "FRED_API_KEY not configured"}), 500
    
    drivers = {}
    
    # CPI (year‑over‑year)
    try:
        resp = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=2",
            timeout=10
        )
        if resp.status_code == 200:
            obs = resp.json().get("observations", [])
            if len(obs) >= 2:
                current = float(obs[0].get("value", 0))
                previous = float(obs[1].get("value", 0))
                drivers['cpi_yoy'] = round(((current / previous) - 1) * 100, 2)
            else:
                drivers['cpi_yoy'] = "N/A"
    except Exception as e:
        drivers['cpi_yoy'] = "N/A"
    
    # Core PCE
    try:
        resp = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id=PCEPI&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1",
            timeout=10
        )
        if resp.status_code == 200:
            obs = resp.json().get("observations", [])
            if obs:
                drivers['core_pce'] = float(obs[0].get("value", 0))
            else:
                drivers['core_pce'] = "N/A"
    except Exception as e:
        drivers['core_pce'] = "N/A"
    
    # 5‑year breakeven inflation
    try:
        resp = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id=T5YIE&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1",
            timeout=10
        )
        if resp.status_code == 200:
            obs = resp.json().get("observations", [])
            if obs:
                drivers['breakeven_5y'] = float(obs[0].get("value", 0))
            else:
                drivers['breakeven_5y'] = "N/A"
    except Exception as e:
        drivers['breakeven_5y'] = "N/A"
    
    return jsonify(drivers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
