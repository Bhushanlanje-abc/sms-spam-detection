
from pathlib import Path
import pandas as pd
import joblib

# Load the trained model
MODEL_PATH = Path("artifacts/spam_classifier.joblib")

if not MODEL_PATH.exists():
    print("Model not found. First run: python train_model.py")
    raise SystemExit

model = joblib.load(MODEL_PATH)

print("=== SMS Spam Detection ===")
print("Enter numeric features about a message.")

num_links = int(input("Number of links: "))
num_words = int(input("Number of words: "))
has_offer = int(input("Contains offer? (1=Yes, 0=No): "))

all_caps = int(input("Mostly capital letters? (1=Yes, 0=No): "))

# Prepare the input in the same format as training data
message = pd.DataFrame([{
    "num_links": num_links,
    "num_words": num_words,
    "has_offer": has_offer,
    
    "all_caps": all_caps,
}])

prediction = int(model.predict(message)[0])
probability = float(model.predict_proba(message)[0][1])

print("\nPrediction:", "SPAM" if prediction == 1 else "HAM / NOT SPAM")
print(f" spam probability: {probability:.1%}")

