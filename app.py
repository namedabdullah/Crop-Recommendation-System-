"""Streamlit front end for the single-feature crop recommender.

Run:  streamlit run app.py   (after `python train.py`)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model/crop_model.joblib")
METRICS_PATH = Path("model/metrics.json")
FEATURE_NAMES = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "Soil pH"}

st.set_page_config(page_title="Crop Recommender", page_icon="🌱", layout="centered")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    return bundle["model"], bundle["feature"], metrics


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

model, feature, metrics = load_artifacts()
feature_label = FEATURE_NAMES[feature]

st.title("🌱 Crop Recommender")
st.write(
    f"You only need to measure **one** soil attribute. "
    f"Our analysis found **{feature_label}** is the most informative, so enter it below."
)

low, high = metrics["feature_range"]
value = st.number_input(
    f"{feature_label} in soil",
    min_value=0.0,
    max_value=float(high) * 1.5,
    value=float(round((low + high) / 2)),
    step=1.0,
    help=f"Values in the training data range from {low:g} to {high:g}.",
)
if not low <= value <= high:
    st.warning(f"{value:g} is outside the training range ({low:g}–{high:g}); the recommendation may be unreliable.")

if st.button("Recommend crop", type="primary"):
    proba = model.predict_proba(pd.DataFrame({feature: [value]}))[0]
    top3 = pd.Series(proba, index=model.classes_).nlargest(3)

    st.success(f"### Best match: **{top3.index[0].title()}**")
    st.write("Top 3 candidates:")
    for crop, p in top3.items():
        st.progress(float(p), text=f"{crop.title()} — {p:.0%}")
    st.caption(
        "A single soil measurement can't separate every crop, so treat this as a shortlist. "
        f"The right crop is in the top 3 about {metrics['test_top3_accuracy']:.0%} of the time."
    )

with st.expander("How was the feature chosen?"):
    scores = pd.Series(metrics["feature_scores_cv_f1"]).rename(index=FEATURE_NAMES)
    st.write("Each feature was used on its own to train a decision tree (5-fold cross-validation, macro F1):")
    st.bar_chart(scores.sort_values(ascending=False))
    col1, col2, col3 = st.columns(3)
    col1.metric("Test accuracy", f"{metrics['test_accuracy']:.0%}")
    col2.metric("Top-3 accuracy", f"{metrics['test_top3_accuracy']:.0%}")
    col3.metric("Random guess", f"{1 / metrics['n_crops']:.0%}")
