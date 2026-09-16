# Loan Default Prediction — ML Engineering Project

A machine learning system that predicts whether a loan applicant is likely to default, served through a REST API and containerized with Docker for fully local deployment.

This project extends a previous data science project (Customer Churn Prediction) into the ML Engineering domain by adding model serving (API) and containerization (Docker) on top of the standard ML pipeline.

## Project Overview

- **Problem:** Binary classification — predict whether a loan applicant will default (`Default = 1`) or not (`Default = 0`)
- **Dataset:** [Loan Default Prediction Dataset](https://www.kaggle.com/) (255,347 rows, 18 columns) — sourced from Kaggle
- **Scope:** Local-only deployment (trained model + API + Docker container running on localhost). No cloud hosting is included in this project's scope.

## Key Results

The dataset is highly imbalanced (88.4% non-default vs. 11.6% default). Five models were trained and compared, with a focus on **Recall** for the default class — the metric that matters most for credit risk assessment, since failing to catch a true default is far more costly than a false alarm.

| Model | Accuracy | Recall (Default) | Precision (Default) | F1-Score (Default) |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 88.5% | 0.03 | 0.60 | 0.06 |
| **Logistic Regression (Balanced)** ✅ | 67.6% | **0.70** | 0.22 | 0.33 |
| Random Forest | 88.5% | 0.03 | 0.63 | 0.06 |
| Random Forest (Balanced) | 88.1% | 0.15 | 0.46 | 0.23 |
| Gradient Boosting (Balanced) | 69.5% | 0.68 | 0.23 | 0.34 |

**Final model: Logistic Regression with `class_weight='balanced'`.** Despite a lower raw accuracy, it achieves by far the best recall for the default class — meaning it catches 70% of applicants who actually default, which is the real business objective. It is also simple, fast, interpretable, and lightweight (ideal for serving via an API).

### Top Predictive Factors (Feature Importance)

Based on the final model's coefficients, the strongest predictors of default are:
1. **Age** — younger applicants are more likely to default
2. **Interest Rate** — higher rates increase default risk
3. **Employment Type (Unemployed)** — strongly increases default risk
4. **Months Employed** — longer employment tenure reduces risk
5. **Income** — higher income reduces risk

## Tech Stack

- **Language:** Python 3.13
- **Data & ML:** pandas, numpy, scikit-learn
- **API:** Flask
- **Containerization:** Docker
- **Model persistence:** joblib

## Project Structure

```
loan-default-prediction-ml/
├── data/
│   └── Loan_default.csv          # raw dataset (not tracked in Docker image)
├── models/                        # saved model artifacts
│   ├── loan_default_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── src/
│   ├── explore_data.py           # full EDA + model comparison (exploratory)
│   ├── train.py                  # clean training script → produces final model
│   ├── app.py                    # Flask API serving the trained model
│   └── templates/
│       └── index.html            # simple web form for manual testing
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## How to Run Locally

### Prerequisites
- Python 3.13
- Docker Desktop (running)

### Option 1: Run with Docker (recommended)

```bash
# Build the image
docker build -t loan-default-api .

# Run the container
docker run -p 5000:5000 loan-default-api
```

The API will be available at `http://127.0.0.1:5000`.

### Option 2: Run locally without Docker

```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python src/train.py

# Run the API
python src/app.py
```

**Note:** Whichever option you use, the server (either the terminal running `python src/app.py`, or the Docker container) must stay running for the API and web form to be reachable — this is a local-only service, not a hosted one.

## Ways to Use the API

### 1. Web Form (easiest — no tools required)

Open `http://127.0.0.1:5000/form` in a browser. Fill in the applicant's details and click **"Check Loan Default Risk"** to get an instant prediction directly on the page.

### 2. `POST /predict` (for programmatic access, e.g. Postman)

Send applicant data as JSON to receive a default prediction.

**Request example:**
```json
{
  "Age": 35,
  "Income": 60000,
  "LoanAmount": 15000,
  "CreditScore": 650,
  "MonthsEmployed": 24,
  "NumCreditLines": 3,
  "InterestRate": 12.5,
  "LoanTerm": 36,
  "DTIRatio": 0.4,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": 1,
  "HasDependents": 0,
  "LoanPurpose": "Auto",
  "HasCoSigner": 0
}
```

**Response example:**
```json
{
  "prediction": "No Default",
  "probability": 0.4651
}
```

## Methodology

1. **Exploratory Data Analysis** — checked for missing values (none found), reviewed value ranges, confirmed class imbalance
2. **Preprocessing** — dropped non-predictive ID column, binary-encoded Yes/No columns, applied One-Hot Encoding (with `drop_first=True`) to multi-category columns
3. **Train/Test Split** — 80/20 split with `stratify` to preserve class balance in both sets
4. **Feature Scaling** — `StandardScaler` fit on training data only (to avoid data leakage), applied to both train and test sets
5. **Model Training & Comparison** — trained and evaluated 5 models, prioritizing recall on the minority (default) class
6. **Model Persistence** — saved the final model, scaler, and feature column order with `joblib`
7. **API Development** — built a Flask REST API that replicates the exact preprocessing pipeline on new input before prediction, plus a simple web form for manual testing
8. **Containerization** — packaged the API into a Docker image for consistent, portable local deployment

## Notes

- This project is scoped for **local deployment only** (no cloud hosting, e.g., Render/Railway/AWS).
- This is a continuation of the author's data science portfolio, following `customer-churn-prediction-ml`, applying the same modeling methodology to a new domain (credit risk) while adding an ML Engineering layer (API + Docker).
