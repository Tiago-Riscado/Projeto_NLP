import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)


def compute_metrics(df: pd.DataFrame, prediction_cols: list) -> pd.DataFrame:
    metrics = []

    for col in prediction_cols:
        valid  = df[df[col].notna()]
        y_true = valid["label_text"]
        y_pred = valid[col]

        print(f"\n{col}")
        print(classification_report(y_true, y_pred, zero_division=0))

        metrics.append({
            "config":          col,
            "accuracy":        accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro":    recall_score   (y_true, y_pred, average="macro", zero_division=0),
            "f1_macro":        f1_score       (y_true, y_pred, average="macro", zero_division=0),
        })

    return pd.DataFrame(metrics)
