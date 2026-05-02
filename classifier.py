import time
import requests
from config import EMOTIONS, HEADERS, API_URL


def normalize_output(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.lower().strip().replace(".", "").replace(",", "").replace("!", "")
    if text in EMOTIONS:
        return text
    for emo in EMOTIONS:
        if f" {emo}" in f" {text}":
            return emo
    return None


def classify(text: str, model: str, prompt: str, retries: int = 3) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "És um especialista em análise de emoções."},
            {"role": "user",   "content": prompt.format(text=text)}
        ],
        "temperature": 0,
        "max_tokens": 15
    }

    for _ in range(retries):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            if r.status_code != 200:
                time.sleep(1)
                continue
            label = normalize_output(r.json()["choices"][0]["message"]["content"])
            if label:
                return label
        except Exception:
            time.sleep(1)

    return None
