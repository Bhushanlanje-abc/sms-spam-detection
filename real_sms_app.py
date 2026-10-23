
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "real_sms_spam_model.joblib"

st.set_page_config(
    page_title="Real SMS Spam Detector",
    page_icon="🛡️",
)

st.title("🛡️ Real SMS Spam Detector")

st.write(
    "Paste an actual SMS, WhatsApp message, or email text "
    "to check whether the model flags it as spam."
)

st.warning(
    "This is a machine-learning screening tool, not a guarantee "
    "that a message is safe or fraudulent."
)

if not MODEL_PATH.exists():
    st.error("The model has not been trained yet.")
    st.code("python train_real_sms_model.py")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

message = st.text_area(
    "Paste your message here",
    height=200,
    placeholder=(
        "Paste the full message you received here..."
    ),
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        # The trained pipeline accepts raw text directly.
        spam_score = float(
            model.predict_proba([message])[0][1]
        )

        prediction = int(
            model.predict([message])[0]
        )

        st.metric(
            "Model spam score",
            f"{spam_score:.1%}",
        )

        if prediction == 1:
            st.error("Result: SPAM-LIKE")
            st.write(
                "The model classifies this message as spam. "
                "Check any links or requests carefully."
            )
        else:
            st.success("Result: NOT FLAGGED AS SPAM")
            st.write(
                "The model did not classify this message as spam. "
                "It could still be a scam."
            )

        st.caption(
            "The score is a model estimate, not a calibrated "
            "probability of fraud. Verify suspicious requests "
            "through the official website or phone number."
        )
