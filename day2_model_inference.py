"""
Day 2 -- Model Inference and Evaluation
Dataset: Loan Prediction (train_u6lujuX_CVtuZ9i.csv / test_Y3wMUE5_7gLdaTN.csv)

Covers:
1. Explore the dataset, identify features/labels
2. Inspect missing values and do basic data preparation
3. Run a pretrained classification model (TabPFN) and generate predictions + confidence
4. Calculate accuracy, precision, recall (+ confusion matrix)
5. Simple CLI interface to display predictions for a new applicant

Model: TabPFN (Prior-Data Fitted Network) is a transformer pretrained once
on millions of synthetic tabular datasets. Its weights are fixed -- "fit"
only loads the training rows as in-context examples, no gradient training
happens here, so this satisfies the "use a pretrained classification
model" requirement for tabular data. Free for non-commercial/learning use;
requires a Prior Labs API key set as the TABPFN_TOKEN environment variable
(get one at https://ux.priorlabs.ai/account after accepting the license).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

TRAIN_PATH = "train_u6lujuX_CVtuZ9i.csv"
TEST_PATH = "test_Y3wMUE5_7gLdaTN.csv"
LABEL_COL = "Loan_Status"
ID_COL = "Loan_ID"

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def explore(df):
    section("1. DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"\nFeatures: {[c for c in df.columns if c not in (ID_COL, LABEL_COL)]}")
    print(f"Label: {LABEL_COL}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nLabel distribution:\n{df[LABEL_COL].value_counts()}")


def inspect_missing(df):
    section("2. MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing if len(missing) else "No missing values found.")
    return missing


def prepare_data(df, encoders=None, fit=True):
    df = df.copy()

    # Numeric: fill with median
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Categorical: fill with mode
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Dependents has a "3+" category -> normalize then encode
    df["Dependents"] = df["Dependents"].replace("3+", "3")

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS + ["Dependents"]:
        if col == "Dependents":
            df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )

    return df, encoders


def plot_confusion_matrix(cm, class_labels, out_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix - Loan Status")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix plot saved to: {out_path}")


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit(
            "Missing TABPFN_TOKEN. Set it with:\n"
            "  export TABPFN_TOKEN=your-key   (bash)\n"
            "  $env:TABPFN_TOKEN='your-key'   (PowerShell)\n"
            "Get a free key at https://ux.priorlabs.ai/account"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    explore(train_df)
    inspect_missing(train_df)

    section("2b. DATA PREPARATION")
    clean_df, encoders = prepare_data(train_df, fit=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(clean_df[LABEL_COL])
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = clean_df[feature_cols]
    print(f"Prepared feature matrix: {X.shape}")
    print("Remaining missing values:", X.isnull().sum().sum())

    section("3. PRETRAINED MODEL (TabPFN) & GENERATE PREDICTIONS")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = TabPFNClassifier(device="cpu", random_state=42)
    # No gradient training: TabPFN's weights are already pretrained.
    # fit() just hands it the training rows to use as in-context examples.
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    preview = pd.DataFrame({
        "actual": label_encoder.inverse_transform(y_val),
        "predicted": label_encoder.inverse_transform(y_pred),
        "confidence": y_proba.max(axis=1).round(3),
    }).reset_index(drop=True)
    print(preview.head(10))

    section("4. EVALUATION METRICS")
    pos_label = list(label_encoder.classes_).index("Y")
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, pos_label=pos_label)
    rec = recall_score(y_val, y_pred, pos_label=pos_label)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision (Y): {prec:.3f}")
    print(f"Recall (Y):    {rec:.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted), labels=", label_encoder.classes_)
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    plot_confusion_matrix(cm, label_encoder.classes_)

    section("5. SIMPLE PREDICTION INTERFACE")
    print("Enter applicant details to get a loan approval prediction.")
    print("(Example values shown in brackets; press Enter to use them.)\n")

    defaults = {
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": "4000", "CoapplicantIncome": "1500",
        "LoanAmount": "120", "Loan_Amount_Term": "360",
        "Credit_History": "1", "Property_Area": "Urban",
    }

    answers = {}
    for field, default in defaults.items():
        raw = input(f"{field} [{default}]: ").strip()
        answers[field] = raw if raw else default

    row = {
        "ApplicantIncome": float(answers["ApplicantIncome"]),
        "CoapplicantIncome": float(answers["CoapplicantIncome"]),
        "LoanAmount": float(answers["LoanAmount"]),
        "Loan_Amount_Term": float(answers["Loan_Amount_Term"]),
        "Credit_History": float(answers["Credit_History"]),
        "Gender": answers["Gender"], "Married": answers["Married"],
        "Dependents": answers["Dependents"].replace("3+", "3"),
        "Education": answers["Education"],
        "Self_Employed": answers["Self_Employed"],
        "Property_Area": answers["Property_Area"],
    }
    input_df = pd.DataFrame([row])
    input_df, _ = prepare_data(input_df, encoders=encoders, fit=False)
    input_X = input_df[feature_cols]

    pred = model.predict(input_X)[0]
    proba = model.predict_proba(input_X)[0]
    label = label_encoder.inverse_transform([pred])[0]
    confidence = proba.max()

    print("\n--- Prediction Result ---")
    print(f"Loan Status: {'APPROVED' if label == 'Y' else 'REJECTED'}")
    print(f"Confidence:  {confidence:.1%}")


if __name__ == "__main__":
    main()
