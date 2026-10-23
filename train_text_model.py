
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sms_spam_text.tsv"
MODEL_PATH = BASE_DIR / "artifacts" / "text_spam_pipeline.joblib"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "sms_spam_text.tsv not found. "
        "Put the extracted UCI dataset in this folder."
    )

# Dataset columns: label and message text
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    on_bad_lines="skip",
)

df = df.dropna(subset=["label", "message"]).copy()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()

df = df[
    df["label"].isin(["ham", "spam"])
    & df["message"].ne("")
].copy()

# ham = 0; spam = 1
X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Convert message words into numerical features, then classify.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("Training the text model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nEvaluation results:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Ham", "Spam"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nTraining complete!")
print("Saved model:", MODEL_PATH)
