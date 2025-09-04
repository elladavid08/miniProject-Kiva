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
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# =========================
# Config (kept from your flow)
# =========================
DEFAULT_FEATURES = ["Country", "Loan Amount", "Sector", "Activity"]

TEST_SIZE = 0.2
VAL_SIZE = 0.2            # portion of TRAIN that becomes validation
RANDOM_STATE = 42

# undersampling: keep paid ~ R * (#defaulted) in the *inner-train* only
PAID_TO_DEFAULTED_RATIO = 2.0

# threshold tuning: maximize F_beta (beta>1 biases recall); we bias recall to avoid missing defaults
F_BETA = 2.0

THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 19), 3)  # thresholds on P(paid)


# =========================
# Helpers
# =========================
def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with Status in {'paid','defaulted'}, map label, and coerce numeric.
    label: 1 = paid, 0 = defaulted
    """
    df = df.copy()

    # Keep only paid & defaulted (drop refunded & others)
    df = df[df["Status"].isin(["paid", "defaulted"])].copy()

    # Map label: paid -> 1, defaulted -> 0
    df["label"] = (df["Status"] == "paid").astype(int)

    # Ensure Loan Amount is numeric
    if "Loan Amount" in df.columns:
        df["Loan Amount"] = pd.to_numeric(df["Loan Amount"], errors="coerce")

    return df


def _undersample_inner_train(X_in: pd.DataFrame, y_in: pd.Series, ratio_paid_to_defaulted: float, seed: int):
    """
    Random undersampling on inner-train:
      - keep all DEFAULTED (y==0)
      - sample PAID (y==1) so that count_paid ~= ratio * count_defaulted
    Returns new X_res, y_res (shuffled).
    """
    rng = np.random.RandomState(seed)

    idx_defaulted = np.where(y_in == 0)[0]
    idx_paid = np.where(y_in == 1)[0]

    n_def = len(idx_defaulted)
    n_paid_target = int(round(ratio_paid_to_defaulted * n_def))

    if n_paid_target >= len(idx_paid):
        # if target >= available, just keep all paid
        chosen_paid = idx_paid
    else:
        chosen_paid = rng.choice(idx_paid, size=n_paid_target, replace=False)

    chosen_idx = np.concatenate([idx_defaulted, chosen_paid])
    rng.shuffle(chosen_idx)

    return X_in.iloc[chosen_idx], y_in.iloc[chosen_idx]


def _build_preprocessor():
    """
    ColumnTransformer for our 4 features:
      - numeric: ["Loan Amount"]
      - categorical: ["Country", "Sector", "Activity"]
    """
    numeric_features = ["Loan Amount"]
    categorical_features = ["Country", "Sector", "Activity"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def _f_beta(precision, recall, beta: float):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (precision * recall) / (b2 * precision + recall + 1e-12)


def _metrics_for_defaulted(y_true, y_pred):
    """
    Return precision/recall/F1 for DEFAULTED class only (label=0).
    """
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average=None, zero_division=0
    )
    return float(p[0]), float(r[0]), float(f[0])


def _pick_best_threshold_on_validation(y_val, p_paid_val, beta: float):
    """
    Scan THRESHOLD_GRID on P(paid) and pick threshold that maximizes F_beta for DEFAULTED.
    Prediction rule: pred_paid = (p_paid >= thr) else defaulted.
    """
    best_thr, best_f = None, -1.0
    best_prec0, best_rec0 = 0.0, 0.0

    for thr in THRESHOLD_GRID:
        y_pred = (p_paid_val >= thr).astype(int)
        p0, r0, _ = _metrics_for_defaulted(y_val, y_pred)
        f = _f_beta(p0, r0, beta)
        if f > best_f:
            best_f, best_thr = f, thr
            best_prec0, best_rec0 = p0, r0

    return best_thr, best_f, best_prec0, best_rec0


def _evaluate_on_test(y_test, p_paid_test, thr: float, model_name: str):
    """
    Produce prints and metric dict for a single model on the TEST set using chosen threshold.
    """
    # AUC (paid positive)
    auc_paid = roc_auc_score(y_test, p_paid_test)
    # AUC if you flip to "defaulted positive" (same numeric value, computed properly)
    auc_defaulted = roc_auc_score(1 - y_test, 1 - p_paid_test)

    # Apply threshold on P(paid) to get labels
    y_pred = (p_paid_test >= thr).astype(int)

    # Reports
    print(f"\n==== Results for model: {model_name} ====")
    print(f"(Evaluated with custom threshold on P(paid) = {thr:.2f})")
    print(f"AUC (paid positive):      {auc_paid:.4f}")
    print(f"AUC (defaulted positive): {auc_defaulted:.4f}\n")

    print("Classification report (target_names=[defaulted, paid]):")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["defaulted", "paid"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_defaulted", "true_paid"],
                         columns=["pred_defaulted", "pred_paid"])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm_df, "\n")

    # Defaulted-only metrics
    p0, r0, f0 = _metrics_for_defaulted(y_test, y_pred)
    print(f"DEFAULTED class → Precision: {p0:.3f} | Recall: {r0:.3f} | F1: {f0:.3f}\n")

    return {
        "auc": float(auc_paid),
        "threshold": float(thr),
        "precision0": float(p0),
        "recall0": float(r0),
        "f1_0": float(f0),
    }


# =========================
# Public API
# =========================
def train_status_models(df: pd.DataFrame):
    """
    Train/validate/test 3 models (LogReg, RF, XGB) for default vs paid on the Kiva data.
    - Removes 'refunded'
    - Splits train/test (TEST_SIZE)
    - Splits inner train/validation (VAL_SIZE)
    - Random-undersamples PAID in inner-train: paid ~= R * defaulted (R=PAID_TO_DEFAULTED_RATIO)
    - Tunes a probability threshold on P(paid) by maximizing F_beta (beta=F_BETA) for DEFAULTED on validation
    - Evaluates on the untouched TEST set with the chosen threshold
    - Prints detailed results and returns a dict with per-model metrics

    Returns
    -------
    dict: {
        "logreg": {"model": Pipeline, "threshold": float, "auc": float, "precision0": float, "recall0": float, "f1_0": float},
        "rf":     {...},
        "xgb":    {...}
    }
    """
    df = _prep_df(df)

    # Features / target
    X = df[DEFAULT_FEATURES].copy()
    y = df["label"].values  # 1=paid, 0=defaulted

    # Basic distributions (after removing refunded)
    full_counts = pd.Series(y).value_counts().rename(index={1: "1 (paid)", 0: "0 (defaulted)"})
    print("Full class distribution (after removing 'refunded'):")
    print(full_counts, "\n")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Train distribution BEFORE undersampling:")
    print(pd.Series(y_train).value_counts().rename(index={1: "1 (paid)", 0: "0 (defaulted)"}), "\n")

    print("Test distribution (unchanged):")
    print(pd.Series(y_test).value_counts().rename(index={1: "1 (paid)", 0: "0 (defaulted)"}), "\n")

    # Inner split (train -> inner-train + val)
    X_tr_in, X_val, y_tr_in, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    # Random undersampling on inner-train
    X_tr_bal, y_tr_bal = _undersample_inner_train(
        X_tr_in, pd.Series(y_tr_in), PAID_TO_DEFAULTED_RATIO, seed=RANDOM_STATE
    )
    print("Inner-train distribution AFTER undersampling:")
    print(pd.Series(y_tr_bal).value_counts().rename(index={1: "1 (paid)", 0: "0 (defaulted)"}), "\n")

    # Preprocessor
    preprocessor = _build_preprocessor()

    # Models
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "xgb": XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss"
        ),
    }

    results = {}

    for name, base_model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clf", base_model)
        ])

        # Fit on undersampled inner-train
        pipe.fit(X_tr_bal, y_tr_bal)

        # Get validation probabilities for P(paid)
        p_paid_val = pipe.predict_proba(X_val)[:, 1]

        # Choose threshold by F_beta (DEFAULTED class focus)
        best_thr, best_f, best_p0, best_r0 = _pick_best_threshold_on_validation(
            y_val, p_paid_val, beta=F_BETA
        )

        print(f"[{name}] best threshold on validation (F{F_BETA}): {best_thr:.3f} | "
              f"F{F_BETA}={best_f:.3f} | P0={best_p0:.3f} | R0={best_r0:.3f}")

        # Evaluate on test with that threshold
        p_paid_test = pipe.predict_proba(X_test)[:, 1]
        metrics_dict = _evaluate_on_test(y_test, p_paid_test, best_thr, name)

        # Save results
        results[name] = {
            "model": pipe,
            **metrics_dict
        }

    # Summary table (focus on defaulted)
    print("==== Summary Table (focus: defaulted class) ====")
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Threshold (Ppaid)": res["threshold"],
            "AUC (defaulted)": res["auc"],  # same numeric value, computed on P(paid)
            "Precision (defaulted)": res["precision0"],
            "Recall (defaulted)": res["recall0"],
            "F1 (defaulted)": res["f1_0"],
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # Also print the chosen thresholds list
    print("\nChosen thresholds by model (P(paid)):")
    for name, res in results.items():
        print(f"  {name}: {res['threshold']:.3f}")

    return results
