import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support
)

# ===== 1) General parameters =====
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "../../OneDrive/מסמכים/הנדסת נתונים/שנה ג/סמסטר ב/סדנת הכנה לפרויקט/data", "bigml.csv")   # <--- your file is in data/bigml.csv

# Selected features / label
FEATURES = ["Country", "Loan Amount", "Sector", "Activity"]
LABEL_COL = "Status"

# Undersampling ratio on TRAIN ONLY: how many "paid" per "defaulted"
PAID_TO_DEFAULTED_RATIO = 2

# Decision threshold on P(paid) used for evaluation (slight recall preference for defaulted)
THRESHOLD_PAID = 0.60


# ===== 2) Load and clean data =====
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Reads CSV, removes 'refunded' rows, and creates a binary label:
    paid -> 1, defaulted -> 0
    """
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Keep only paid/defaulted
    df = df[df[LABEL_COL].str.lower().isin(["paid", "defaulted"])].copy()

    # Binary label
    df["label"] = (df[LABEL_COL].str.lower() == "paid").astype(int)

    # Ensure numeric dtype for Loan Amount
    if not np.issubdtype(df["Loan Amount"].dtype, np.number):
        df["Loan Amount"] = pd.to_numeric(df["Loan Amount"], errors="coerce")

    return df


# ===== 3) Undersampling (TRAIN only) =====
def undersample(df: pd.DataFrame, ratio_paid_to_defaulted: int = 2) -> pd.DataFrame:
    """
    Keep all defaulted, sample paid according to the given ratio.
    Expects a DataFrame that already contains the 'label' column.
    """
    df_defaulted = df[df["label"] == 0]
    df_paid = df[df["label"] == 1]

    n_def = len(df_defaulted)
    n_paid_sample = min(len(df_paid), n_def * ratio_paid_to_defaulted)

    df_paid_sample = df_paid.sample(n=n_paid_sample, random_state=RANDOM_STATE)
    df_balanced = pd.concat([df_defaulted, df_paid_sample], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    return df_balanced.reset_index(drop=True)


# ===== 4) Preprocessing =====
def build_preprocessor():
    """
    Returns a ColumnTransformer:
    - Numeric: impute + scale
    - Categorical: impute + one-hot encode
    """
    numeric_features = ["Loan Amount"]
    categorical_features = ["Country", "Sector", "Activity"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # with_mean=False for sparse support
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


# ===== 5) Models =====
def get_models():
    """
    Returns a dictionary of models to train and compare.
    """
    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=400,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist"
        ),
    }
    return models


# ===== 6) Build pipeline (preprocessing + model) =====
def build_pipeline(model):
    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", model)
    ])
    return pipe


# ===== 7) Evaluation helpers =====
def evaluate_model(model_name: str, y_true, y_pred, y_proba_paid, threshold_used: float | None = None):
    """
    Prints detailed evaluation:
    - AUC (paid & defaulted as positive)
    - Classification report
    - Confusion Matrix
    - Precision/Recall/F1 for defaulted (label=0)
    """
    y_proba_defaulted = 1.0 - y_proba_paid
    auc_paid = roc_auc_score(y_true, y_proba_paid)
    auc_defaulted = roc_auc_score(1 - y_true, y_proba_defaulted)

    print(f"\n==== Results for model: {model_name} ====")
    if threshold_used is not None:
        print(f"(Evaluated with custom threshold on P(paid) = {threshold_used:.2f})")
    print(f"AUC (paid positive):      {auc_paid:.4f}")
    print(f"AUC (defaulted positive): {auc_defaulted:.4f}\n")

    print("Classification report (target_names=[defaulted, paid]):")
    print(classification_report(y_true, y_pred, target_names=["defaulted", "paid"]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["true_defaulted", "true_paid"], columns=["pred_defaulted", "pred_paid"]))

    # Metrics for defaulted (label=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None)
    print(f"\nDEFAULTED class → Precision: {precision[0]:.3f} | Recall: {recall[0]:.3f} | F1: {f1[0]:.3f}\n")


def evaluate_model_summary(model_name: str, y_true, y_pred, y_proba_paid, results_list: list, threshold_used: float | None):
    """
    Collects model performance into results_list for summary table.
    Stores: model, threshold, AUC(defaulted), Precision/Recall/F1(defaulted).
    """
    y_proba_defaulted = 1.0 - y_proba_paid
    auc_defaulted = roc_auc_score(1 - y_true, y_proba_defaulted)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None)

    results_list.append({
        "Model": model_name,
        "Threshold (Ppaid)": threshold_used if threshold_used is not None else 0.50,
        "AUC (defaulted)": round(auc_defaulted, 4),
        "Precision (defaulted)": round(precision[0], 3),
        "Recall (defaulted)": round(recall[0], 3),
        "F1 (defaulted)": round(f1[0], 3),
    })


# ===== 8) main =====
def main():
    # Load full data (paid/defaulted only)
    df = load_and_clean(CSV_PATH)

    # Show full distribution (real-world after removing refunded)
    print("Full class distribution (after removing 'refunded'):")
    print(df["label"].value_counts())

    # Split BEFORE any sampling; keep test as-is (realistic evaluation)
    X = df[FEATURES].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print("\nTrain distribution BEFORE undersampling:")
    print(y_train.value_counts())
    print("\nTest distribution (unchanged):")
    print(y_test.value_counts())

    # Build a TRAIN DataFrame with label for undersampling
    df_train = X_train.copy()
    df_train["label"] = y_train.values

    # Undersample TRAIN only
    df_train_bal = undersample(df_train, ratio_paid_to_defaulted=PAID_TO_DEFAULTED_RATIO)
    print("\nTrain distribution AFTER undersampling:")
    print(df_train_bal["label"].value_counts())

    # Re-split X_train_bal / y_train_bal
    X_train_bal = df_train_bal[FEATURES].copy()
    y_train_bal = df_train_bal["label"].copy()

    # Prepare models
    models = get_models()
    results = []

    for name, clf in models.items():
        pipe = build_pipeline(clf)

        # Fit on undersampled TRAIN
        pipe.fit(X_train_bal, y_train_bal)

        # Probabilities for paid (positive class)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_proba_paid = pipe.predict_proba(X_test)[:, 1]
        else:
            # fallback: use decision_function normalized to [0,1]
            decision = pipe.decision_function(X_test)
            y_proba_paid = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)

        # Apply custom threshold on P(paid)
        y_pred_thresh = (y_proba_paid >= THRESHOLD_PAID).astype(int)

        # Detailed evaluation (with threshold info)
        evaluate_model(name, y_test, y_pred_thresh, y_proba_paid, threshold_used=THRESHOLD_PAID)

        # Summary row
        evaluate_model_summary(name, y_test, y_pred_thresh, y_proba_paid, results, threshold_used=THRESHOLD_PAID)

    # Print summary table
    print("\n==== Summary Table (focus: defaulted class) ====")
    df_results = pd.DataFrame(results)
    # Order columns nicely
    df_results = df_results[["Model", "Threshold (Ppaid)", "AUC (defaulted)", "Precision (defaulted)", "Recall (defaulted)", "F1 (defaulted)"]]
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
