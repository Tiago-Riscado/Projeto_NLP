import requests
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# CONFIGURAÇÃO

OPENROUTER_API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "PLN-TrabalhoFinal"
}

INPUT_CSV = r"C:\Users\tiago\Desktop\tweets.csv"
OUTPUT_DIR = r"D:\Universidade\Lic_IACD\2025-2026\NLP\Projeto\Resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

EMOTIONS = ["sadness", "joy", "love", "anger", "fear"]

LABEL_COLUMN_VALUES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear"
}

MODELS = {
    "llama31": "meta-llama/llama-3.1-8b-instruct",
    "mistral7b": "mistralai/mistral-7b-instruct"
}

# PROMPTS

PROMPTS = {
    "simples": """Classifica a emoção principal do texto. 
Responde apenas com:
sadness, joy, love, anger ou fear.

Texto: "{text}"
""",
    "avançado": """Analisa cuidadosamente a emoção expressa no texto.
Considera sarcasmo, emojis e linguagem informal.

Emoções possíveis:
sadness, joy, love, anger, fear

Responde apenas com uma palavra.

Texto: "{text}"
""",
    "conservador": """Classifica a emoção dominante do texto.
Em caso de dúvida, escolhe a emoção principal.

Responde apenas com:
sadness, joy, love, anger ou fear.

Texto: "{text}"
"""
}

# FUNÇÕES

# Normaliza a saída do modelo para uma das emoções válidas
def normalize_output(text):
    if not isinstance(text, str):
        return None
    text = text.lower().strip()
    text = text.replace(".", "").replace(",", "").replace("!", "")
    if text in EMOTIONS:
        return text
    for emo in EMOTIONS:
        if f" {emo}" in f" {text}":
            return emo
    return None

# Classifica o texto usando o modelo e prompt fornecidos
def classify(text, model, prompt, retries=3):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "És um especialista em análise de emoções."},
            {"role": "user", "content": prompt.format(text=text)}
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

            out = r.json()["choices"][0]["message"]["content"]
            label = normalize_output(out)
            if label:
                return label

        except Exception:
            time.sleep(1)

    return None

# DADOS

df = pd.read_csv(INPUT_CSV)

# remover labels inválidos
df = df[df[LABEL_COLUMN].isin(LABEL_COLUMN_VALUES.keys())].copy()

df["label_text"] = df[LABEL_COLUMN].map(LABEL_COLUMN_VALUES)

df = df.sample(10, random_state=42).reset_index(drop=True)

# CLASSIFICAÇÃO

prediction_cols = []

for model_name, model_id in MODELS.items():
    for prompt_name, prompt in PROMPTS.items():

        col = f"{model_name}_{prompt_name}"
        print(f"\n A classificar: {col}")

        df[col] = [
            classify(str(t), model_id, prompt)
            for t in tqdm(df[TEXT_COLUMN], desc=col)
        ]

        prediction_cols.append(col)

df.to_csv(os.path.join(OUTPUT_DIR, "predictions_all.csv"), index=False)

# AVALIAÇÃO

metrics = []

for col in prediction_cols:

    valid = df[df[col].notna()]

    y_true = valid["label_text"]
    y_pred = valid[col]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{col}")
    print(classification_report(y_true, y_pred, zero_division=0))

    metrics.append({
        "config": col,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

print("\n Concluído com sucesso")
print("Resultados em:", OUTPUT_DIR)
