"""Streamlit interface for the raw-URL phishing classifier."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from train import FEATURE_NAMES, extract_features, normalise_url

ARTIFACTS = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="URL Risk Check", page_icon="🔎")
st.title("URL phishing risk check")
st.caption("A URL-string-only research model. It is not a guarantee that a link is safe or malicious.")

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "raw_url_extra_trees.joblib")

url = st.text_input(
    "URL to check",
    placeholder="https://example.com/account/login",
    help="Include http:// or https://. The protocol is a model feature, so the app will not guess it.",
)
if st.button("Assess URL", type="primary"):
    if not url.strip():
        st.warning("Enter a URL first.")
    elif "://" not in url.strip():
        st.warning("Include the protocol, for example https://www.google.com. The app will not assume HTTP or HTTPS.")
    else:
        try:
            model = load_model()
            row = pd.DataFrame([extract_features(url)], columns=FEATURE_NAMES)
            phishing_probability = float(model.predict_proba(row)[0][list(model.classes_).index(0)])
            prediction = int(model.predict(row)[0])
            if prediction == 0:
                st.error(f"Potential phishing — model score: {phishing_probability:.1%}")
            else:
                st.success(f"Likely legitimate — phishing score: {phishing_probability:.1%}")
            with st.expander("Derived URL features"):
                st.caption(f"Normalized input: {normalise_url(url)}")
                st.dataframe(row.T.rename(columns={0: "value"}))
        except FileNotFoundError:
            st.error("Model artifacts are missing. Run train.py first.")

with st.expander("Important limitations"):
    st.markdown("""
    - The model sees only URL characters; it does not inspect the live website, reputation, or page content.
    - A legitimate-looking result is not a safety endorsement. Verify unexpected links through trusted channels.
    - Scores reflect this dataset and its time period; test on independent, recent URLs before operational use.
    """)

if (ARTIFACTS / "metrics.json").exists():
    with st.expander("Validation summary"):
        st.json(json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8")))
