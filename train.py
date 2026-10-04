"""Find the single best soil feature for predicting the crop, then train a model on it.

Run:  python train.py
Outputs:  model/crop_model.joblib  and  model/metrics.json
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("data/soil_measures.csv")
MODEL_DIR = Path("model")
FEATURES = ["N", "P", "K", "ph"]
TARGET = "crop"
RANDOM_STATE = 42


def make_model():
    # A shallow tree splits one numeric feature into value ranges -> crops.
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE)


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE
    )

    # 1. Feature selection: score each feature on its own with 5-fold CV (training data only).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    feature_scores = {}
    for feature in FEATURES:
        scores = cross_val_score(make_model(), X_train[[feature]], y_train, cv=cv, scoring="f1_macro")
        feature_scores[feature] = round(scores.mean(), 4)
        print(f"{feature:>3}: CV macro F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_feature = max(feature_scores, key=feature_scores.get)
    print(f"\nBest single feature: {best_feature}")

    # 2. Train the final model on the best feature and evaluate on the held-out test set.
    model = make_model().fit(X_train[[best_feature]], y_train)
    y_pred = model.predict(X_test[[best_feature]])
    y_proba = model.predict_proba(X_test[[best_feature]])

    metrics = {
        "best_feature": best_feature,
        "feature_scores_cv_f1": feature_scores,
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "test_top3_accuracy": round(top_k_accuracy_score(y_test, y_proba, k=3, labels=model.classes_), 4),
        "n_crops": int(df[TARGET].nunique()),
        "feature_range": [float(df[best_feature].min()), float(df[best_feature].max())],
    }
    print(f"Test accuracy:       {metrics['test_accuracy']:.3f}")
    print(f"Test macro F1:       {metrics['test_f1_macro']:.3f}")
    print(f"Test top-3 accuracy: {metrics['test_top3_accuracy']:.3f}")
    print(f"(Random guess over {metrics['n_crops']} crops = {1 / metrics['n_crops']:.3f})")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature": best_feature}, MODEL_DIR / "crop_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
