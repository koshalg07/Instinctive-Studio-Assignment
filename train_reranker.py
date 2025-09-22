import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib


def load_training_pairs(features_jsonl: str):
    X, y = [], []
    with open(features_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") is None:
                continue
            X.append([row["vector_score"], row["bm25_score"]])
            y.append(int(row["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("features_jsonl", help="Path to training features jsonl (from features.save_features_for_questions)")
    parser.add_argument("--out", default="data/reranker_lr.joblib", help="Where to save the trained model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    X, y = load_training_pairs(args.features_jsonl)
    if len(y) == 0:
        raise SystemExit("No labeled data found in features file.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation ROC-AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()


