
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "text_spam_pipeline.joblib"

st.set_page_config(
    page_title="Text Spam Analyzer",
    page_icon="🛡️",
)

st.title("🛡️ Text Spam Analyzer")

st.write(
    "Paste an SMS or message below to check whether "
    "the ML model classifies it as spam-like."
)

st.info(
    "This is a research prototype trained on an English SMS dataset. "
    "It cannot guarantee that a message is safe or identify every scam."
)

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.code("python train_text_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

message = st.text_area(
    "Paste your message here",
    height=180,
    placeholder=(
        "Example: Congratulations! Click this link "
        "to claim your reward..."
    ),
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        prediction = int(model.predict([message])[0])
        spam_score = float(model.predict_proba([message])[0][1])

        if prediction == 1:
            st.error("Result: SPAM-LIKE / POTENTIALLY SUSPICIOUS")
        else:
            st.success("Result: HAM-LIKE / NOT FLAGGED AS SPAM")

        st.metric(
            "Model-estimated spam score",
            f"{spam_score:.1%}",
        )

        st.caption(
            "This score is a model estimate, not a verified probability "
            "that the message is a scam. Do not use it alone to decide "
            "whether a message or link is safe."
        )
