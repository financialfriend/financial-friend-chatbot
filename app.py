from fastapi import FastAPI
from pydantic import BaseModel
import requests
import pandas as pd
import re
from thefuzz import fuzz
import yfinance as yf
from bs4 import BeautifulSoup
import uvicorn
import os
from sentence_transformers import SentenceTransformer, util

# Load from environment if on Render
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")

# Load instruction model and instructions
instruction_model = SentenceTransformer('all-MiniLM-L6-v2')

INSTRUCTIONS = [
    "Never mention, reference, or promote any website except https://financialfriend.in.",
    "Never ask the user to visit another site or app for financial data.",
    "Must use friendly emojis in every answer.",
    "Only provide to the point answers no other suggestions or knowledge.",
    "It is your priority to do risk profiling based on the 3 questions mentioned below when user ask you that he wants to invest (ask questions one by one and no need to provide their weightage) and only provide the category to which they come under and their return expectation.",
    "How would you react if your portfolio dropped by 20%?",
    "(a) Invest more - 30",
    "(b) Stay calm and wait - 24",
    "(c) Feel nervous but wait - 18",
    "(d) Sell part of the investment - 9",
    "(e) Sell fully - 0",
    "What best describes your experience and behavior with investing?",
    "(a) 10+ years in equities; invest regularly, see risk as opportunity - 35",
    "(b) 5–10 years; mostly disciplined; risk is necessary - 28",
    "(c) 2–5 years; sometimes miss goals; risk must be managed - 21",
    "(d) <2 years; invest irregularly; avoid risk - 10",
    "(e) No experience; no regular habit; want no risk - 0",
    "How would you describe your investment goals and liquidity needs?",
    "(a) 15+ year goal, 15%+ return expected, emergency fund in place - 35",
    "(b) 10–15 years, expect 12–15%, would withdraw only in rare cases - 28",
    "(c) 5–10 years, expect 10–12%, may withdraw if needed - 21",
    "(d) 3–5 years, expect 7–10%, may partially withdraw - 10",
    "(d) <3 years, expect <7%, likely to redeem fully - 0",
    "Scoring Interpretation",
    "81–100 Very High Risk Tolerance (Aggressive Investor) >16%",
    "61–80 High Risk Tolerance (Growth-Oriented Investor) 14–16%",
    "41–60 Moderate Risk Tolerance (Balanced Investor) 10–14%",
    "21–40 Low Risk Tolerance (Conservative Investor) 8–10%",
    "0–20 Very Low Risk Tolerance (Cautious Investor) 6–8%",
    "Always end with: For personalized help, visit https://financialfriend.in or WhatsApp us at https://bit.ly/4kf1P8f 💬",
    "Always respond with numeric facts and clear figures, not vague descriptions.",
    "Focus only on relevant financial details like stock prices, changes, highs, lows, and historical comparisons — no volume data.",
    "Must follow all the guidelines mentioned.",
    "Only promote our Financial Planner to User."
]

SYSTEM_PROMPT = "You are a friendly financial advisor from https://financialfriend.in 💼. Always use emojis 😄, give numeric answers where possible, and redirect users to our site. Your tone is professional, yet helpful."

def get_relevant_instructions(query, k=5):
    embeddings = instruction_model.encode(INSTRUCTIONS + [query])
    scores = util.cos_sim(embeddings[-1], embeddings[:-1])
    top_idxs = scores[0].argsort()[-k:][::-1]
    return "\n".join([INSTRUCTIONS[i] for i in top_idxs])

def normalize(text):
    text = text.lower()
    text = re.sub(r'\bltd\b|\blimited\b|[\.\,\-\(\)]', '', text)
    return text.strip()

def get_nse_symbol(company_name):
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
    except Exception:
        return None
    company_name = normalize(company_name)
    best_score = 0
    best_match = None
    for _, row in df.iterrows():
        score = fuzz.partial_ratio(company_name, normalize(str(row["NAME OF COMPANY"])))
        if score > best_score:
            best_score = score
            best_match = row
    return best_match["SYMBOL"] if best_match and best_score >= 70 else None

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(f"{symbol}.NS")
        data = stock.info
        return (
            f"📊 Stock Analysis for {symbol}:\n"
            f"🔹 Current Price: ₹{data.get('currentPrice')}\n"
            f"🔹 Day High: ₹{data.get('dayHigh')}\n"
            f"🔹 Day Low: ₹{data.get('dayLow')}\n"
            f"🔹 Previous Close: ₹{data.get('previousClose')}"
        )
    except:
        return "⚠️ Unable to fetch stock data."

def scrape_fd_rates():
    try:
        res = requests.get("https://www.bankbazaar.com/fixed-deposit/sbi-fixed-deposit-rate.html", headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table')
        if not table:
            return "⚠️ SBI FD rates table not found."
        rows = table.find_all('tr')
        result = []
        for row in rows:
            cols = row.find_all(['td', 'th'])
            result.append(" | ".join(col.text.strip() for col in cols))
        return "\n".join(result[:6])
    except:
        return "⚠️ Error fetching FD rates."

class UserInput(BaseModel):
    message: str
    history: list

app = FastAPI()

@app.post("/chat")
def chat_with_bot(payload: UserInput):
    msg = payload.message
    history = payload.history or []

    if any(k in msg.lower() for k in ["stock", "share", "price"]):
        if len(msg.split()) <= 5:
            return {"response": "🤔 Please provide the company name."}
        symbol = get_nse_symbol(msg)
        if symbol:
            stock_summary = get_stock_data(symbol)
            msg += f"\n\n{stock_summary}"
        else:
            msg += "\n\n⚠️ Company not found."

    if "fd" in msg.lower():
        fd = scrape_fd_rates()
        return {"response": f"🏦 SBI FD Rates:\n{fd}\n\n📞 Contact +91 XXXXXXXXXX or visit https://financialfriend.in 💼"}

    relevant_instructions = get_relevant_instructions(msg)
    prompt = SYSTEM_PROMPT + "\n" + relevant_instructions

    messages = [{"role": "system", "content": prompt}]
    for u, b in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": msg})

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 700
        }
    )
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
    else:
        content = f"⚠️ API Error: {response.text}"
    return {"response": content}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)