import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "PLN-TrabalhoFinal"
}

INPUT_CSV  = os.getenv("INPUT_CSV",  "tweets.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "Resultados")

TEXT_COLUMN  = "text"
LABEL_COLUMN = "label"

EMOTIONS = ["sadness", "joy", "love", "anger", "fear"]

LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear"}

MODELS = {
    "llama31":   "meta-llama/llama-3.1-8b-instruct",
    "mistral7b": "mistralai/mistral-7b-instruct"
}
