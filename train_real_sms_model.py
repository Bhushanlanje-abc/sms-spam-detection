
from pathlib import Path
import json
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import pandas as pd
import joblib
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "real_sms_spam_model.joblib"


def load_huggingface_dataset(repo_id):
    # Special handling for the Hinglish dataset because
    # its JSON structure does not load through the normal
    # Hugging Face split loader.

    if repo_id == "bolewara/hinglish-scam-text-dataset":
        possible_files = [
            "data.json",
            "hinglish-scam-text-dataset/data.json",
        ]

        json_path = None

        for filename in possible_files:
            try:
                json_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
                break
            except EntryNotFoundError:
                continue

        if json_path is None:
            raise FileNotFoundError(
                "Could not find data.json in the Hinglish dataset. "
                "Check the dataset's Files tab on Hugging Face."
            )

        with open(json_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        # The dataset stores records inside a "rows" array.
        if isinstance(raw_data, dict) and "rows" in raw_data:
            rows = raw_data["rows"]
        elif isinstance(raw_data, list):
            rows = raw_data
        else:
            raise ValueError(
                "Unexpected JSON format in the Hinglish dataset."
            )

        df = pd.DataFrame(rows)

        print(
            f"Loaded {len(df)} Hinglish dataset rows."
        )

        return df

    # Normal loader for datasets such as scam-ham-india.
    print(f"Loading dataset: {repo_id}")

    dataset = load_dataset(repo_id, split="train")
    df = dataset.to_pandas()

    print(f"Loaded {len(df)} rows from {repo_id}")

    return df


def normalize_dataset(df, source):
    """Convert supported message and label formats to a common schema."""

    text_column = next(
        (
            col for col in ["text", "message", "sms"]
            if col in df.columns
        ),
        None,
    )

    label_column = next(
        (
            col for col in ["label", "target", "class"]
            if col in df.columns
        ),
        None,
    )

    if text_column is None or label_column is None:
        raise ValueError(
            f"Cannot identify text/label columns in {source}. "
            f"Available columns: {df.columns.tolist()}"
        )

    result = df[[text_column, label_column]].copy()
    result.columns = ["text", "label"]

    def convert_label(value):
        label = str(value).strip().lower()

        if label in {
            "1", "spam", "scam", "fraud", "phishing",
            "malicious", "unsafe"
        }:
            return 1

        if label in {
            "0", "ham", "legitimate", "legit",
            "benign", "safe", "not spam", "not_spam"
        }:
            return 0

        return None

    result["label"] = result["label"].map(convert_label)
    result["text"] = result["text"].astype("string").str.strip()

    result = result.dropna(subset=["text", "label"])
    result = result[result["text"].ne("")].copy()
    result["label"] = result["label"].astype(int)

    return result


# 1. Download India-focused datasets.
print("Downloading Indian scam datasets...")

india_df = load_huggingface_dataset(
    "anmolshrivastav/scam-ham-india"
)

hinglish_df = load_huggingface_dataset(
    "bolewara/hinglish-scam-text-dataset"
)

# 2. Normalize both datasets.
india_df = normalize_dataset(
    india_df, "Indian scam/spam dataset"
)

hinglish_df = normalize_dataset(
    hinglish_df, "English/Hinglish scam dataset"
)

# 3. Combine and remove exact duplicate messages.
df = pd.concat(
    [india_df, hinglish_df],
    ignore_index=True,
)

df["text_key"] = (
    df["text"].str.lower().str.replace(r"\s+", " ", regex=True)
)

# If duplicates have conflicting labels, discard those messages.
conflicts = df.groupby("text_key")["label"].nunique()
conflicting_texts = conflicts[conflicts > 1].index

df = df[~df["text_key"].isin(conflicting_texts)]
df = df.drop_duplicates(subset="text_key").copy()

df = df[["text", "label"]].reset_index(drop=True)

print("\nCombined dataset size:", len(df))
print("\nLabel distribution:")
print(df["label"].map({0: "Legitimate", 1: "Scam"}).value_counts())

if df["label"].nunique() != 2:
    raise ValueError("Both legitimate and scam labels are required.")

# 4. Split messages for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

# 5. Build TF-IDF + Logistic Regression pipeline.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

# 6. Train and evaluate.
print("\nTraining the model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nEVALUATION")
print("Accuracy :", accuracy_score(y_test, predictions))
print(
    "Precision:",
    precision_score(y_test, predictions, zero_division=0),
)
print(
    "Recall   :",
    recall_score(y_test, predictions, zero_division=0),
)
print(
    "F1-score :",
    f1_score(y_test, predictions, zero_division=0),
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Scam"],
    zero_division=0,
))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 7. Save the model.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Training finished!")
