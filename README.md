# SMS Spam Detection

A beginner-friendly machine learning project that aims to identify potentially suspicious or unwanted SMS messages using Python.

## Project Overview

This project explores how machine learning and text-processing techniques can classify SMS messages as spam/scam or legitimate.

The goal is to help users recognize suspicious messages, including messages that may involve fake offers, payment requests, or fraudulent links.

## Features

- SMS text classification prototype
- Text preprocessing and TF-IDF feature extraction
- Machine learning with Logistic Regression
- Streamlit interface for testing messages
- Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix

*Note: Features above describe the intended model workflow. Verify that each feature is implemented and working in the current version before treating the project as complete.*

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Hugging Face Datasets
- Jupyter Notebook

## Project Structure

```text
.
├── train_real_sms_model.py
├── real_sms_app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── spam_detection_dataset.csv
├── Spam_DetectionLinear_Regression.ipynb
├── .gitignore
└── README.md
```

This is an example structure. Adjust it to match the files actually present in the repository.

## Getting Started

### 1. Install Python

Install a compatible Python version from https://www.python.org/downloads/.

### 2. Download the project

On the GitHub repository page, select **Code → Download ZIP**, then extract the folder.

Alternatively, run:

```bash
git clone https://github.com/YOUR-USERNAME/sms-spam-detection.git
cd sms-spam-detection
```

Replace `YOUR-USERNAME` with the repository owner's GitHub username.

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Train the text model

```bash
python train_real_sms_model.py
```

### 6. Start the Streamlit app

After model training succeeds:

```bash
python -m streamlit run real_sms_app.py
```

Streamlit should provide a local address to open in your browser.

## Current Status

This project is under development. The text-model training script currently needs a dataset-loading issue resolved before the complete training-to-app workflow can be verified.

## Limitations

- Predictions are estimates, not proof that a message is fraudulent or safe.
- The quality of predictions depends on the dataset and model evaluation.
- New scam patterns and Hindi/Hinglish messages may not be recognized reliably.
- Do not use this prototype as the only basis for financial or security decisions.

## Future Improvements

- Improve Hindi and Hinglish message coverage.
- Add more diverse, carefully checked training examples.
- Test the model on messages excluded from training.
- Improve the interface and add example test messages.
- Deploy a publicly accessible demo after testing.

## Disclaimer

This project is for educational and research purposes. It does not guarantee the detection of every spam or fraudulent message.


## Author : Bhushan Lanje
