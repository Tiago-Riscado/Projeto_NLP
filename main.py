import os
import pandas as pd
from tqdm import tqdm

from config  import INPUT_CSV, OUTPUT_DIR, TEXT_COLUMN, LABEL_COLUMN, LABEL_MAP, MODELS
from prompts import PROMPTS
from classifier import classify
from evaluate   import compute_metrics

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar e preparar dados
df = pd.read_csv(INPUT_CSV)
df = df[df[LABEL_COLUMN].isin(LABEL_MAP.keys())].copy()
df["label_text"] = df[LABEL_COLUMN].map(LABEL_MAP)
df = df.sample(10, random_state=42).reset_index(drop=True)

# Classificação
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

# Avaliação
metrics_df = compute_metrics(df, prediction_cols)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

print("\n Concluído com sucesso")
print("Resultados em:", OUTPUT_DIR)
