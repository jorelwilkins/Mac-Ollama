#!/bin/bash
# =============================
# All-in-one setup for Ollama + SD + Multi-Variation Images
# For macOS M1/M2/M3
# =============================

set -e

echo "=== Ollama + Stable Diffusion Multi-Variation Setup ==="

# --- Step 1: Create Python virtual environment ---
echo "Creating Python virtual environment at ~/ollama_env..."
python3 -m venv ~/ollama_env
source ~/ollama_env/bin/activate

# --- Step 2: Upgrade pip ---
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Step 3: Install required Python packages ---
echo "Installing Python packages..."
pip install diffusers transformers accelerate safetensors \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
            langchain-ollama langchain-community yfinance requests

# --- Step 4: Pull Ollama model ---
echo "Pulling Ollama LLaMA model..."
read -p "Enter model name to pull (llama3:8b / llama3:70b): " MODEL_NAME
ollama pull $MODEL_NAME

# --- Step 5: Create the multi-variation image script ---
SCRIPT_PATH=~/ollama_env/ollama_image_agent_multi.py
echo "Creating Python script at $SCRIPT_PATH"
cat << 'EOF' > $SCRIPT_PATH
import os
import requests
import yfinance as yf
from langchain_ollama import ChatOllama
from diffusers import StableDiffusionPipeline
import torch

# ---------- CONFIG ----------
GNEWS_API_KEY = "4f3f7f208ddafe83ab7af6db56549ab6"
IMAGE_FOLDER = "generated_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ---------- FUNCTION DEFINITIONS ----------
def get_live_news(max_articles=5, topic=None):
    q = f"&q={requests.utils.quote(topic)}" if topic else ""
    url = f"https://gnews.io/api/v4/top-headlines?token={GNEWS_API_KEY}&lang=en{q}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        articles = data.get("articles", [])
        return "\n".join([f'- {a.get("title","No title")} ({a.get("source",{}).get("name","Unknown")})' for a in articles[:max_articles]]) \
            if articles else "No news available"
    except Exception as e:
        return f"News fetch error: {e}"

def get_weather(location):
    try:
        url = f"https://wttr.in/{requests.utils.quote(location)}?format=j1"
        j = requests.get(url, timeout=8).json()
        cur = j.get("current_condition", [{}])[0]
        return f"{location} — {cur.get('temp_F')}°F, {cur.get('weatherDesc',[{}])[0].get('value','')}"
    except Exception as e:
        return f"Weather fetch error: {e}"

def get_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        price = info.get("regularMarketPrice")
        return f"{symbol}: ${price}" if price else f"No data for {symbol}"
    except Exception as e:
        return f"Stock fetch error: {e}"

def get_sports(league="nfl"):
    league_map = {"nba":"basketball/nba","nfl":"football/nfl","mlb":"baseball/mlb","nhl":"hockey/nhl"}
    path = league_map.get(league.lower())
    if not path: return "League not supported."
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
        j = requests.get(url, timeout=8).json()
        events = j.get("events", [])
        lines = []
        for e in events[:10]:
            comp = e.get("competitions",[{}])[0]
            status = comp.get("status",{}).get("type",{}).get("description","")
            teams = comp.get("competitors",[])
            teams_line = " vs ".join([f"{t.get('team',{}).get('displayName')} ({t.get('homeAway')}) {t.get('score','')}" for t in teams])
            lines.append(f"{teams_line} — {status}")
        return "\n".join(lines) if lines else f"No {league.upper()} games today."
    except Exception as e:
        return f"Sports fetch error: {e}"

# ---------- INIT MODELS ----------
MODEL_NAME = input("Choose Ollama model for chat (llama3:8b / llama3:70b): ").strip()
llm = ChatOllama(model=MODEL_NAME)

print("\n🗨️ Ollama Unified Agent — type 'exit' to quit\nCommands: news, weather <loc>, stock <sym>, sports <league>, image <variations>, or normal questions")

# Load Stable Diffusion once for multi-variation image generation
print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("mps")

while True:
    usr = input("\nYou: ").strip()
    if usr.lower() in ["exit","quit"]: break
    parts = usr.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts)>1 else ""
    
    # Live data
    context = ""
    if cmd == "news":
        context = get_live_news(topic=arg)
    elif cmd == "weather":
        context = get_weather(arg if arg else "Montclair,NJ")
    elif cmd == "stock":
        context = get_stock(arg.upper())
    elif cmd == "sports":
        context = get_sports(arg.lower() if arg else "nfl")
    elif cmd == "image":
        try:
            variations = int(arg) if arg else 1
            prompt_request = f"Generate a detailed text prompt for image generation: {usr}"
            img_prompt = llm.invoke(prompt_request).content.strip()
            for i in range(max(1, min(variations,5))):
                img = pipe(img_prompt).images[0]
                fname = f"{IMAGE_FOLDER}/{usr.replace(' ','_')}_var{i+1}.png"
                img.save(fname)
                print(f"✅ Image saved: {fname}")
            continue
        except Exception as e:
            print(f"Image generation error: {e}")
            continue

    # Build prompt for chat
    prompt_text = f"Live context:\n{context}\nUser question: {usr}" if context else usr
    resp = llm.invoke(prompt_text)
    print("\n🤖 Ollama:\n", resp.content)
EOF

# --- Step 6: Instructions ---
echo "✅ Setup complete!"
echo "To run your Ollama Unified Agent:"
echo "1️⃣ source ~/ollama_env/bin/activate"
echo "2️⃣ python ~/ollama_env/ollama_image_agent_multi.py"
echo "Supports: chat, news, weather, stocks, sports, multi-variation images"
