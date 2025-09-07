import pandas as pd
import numpy as np
import io, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from time import perf_counter


def train_repayment_time_model(df, features):
    """
    Train models to predict repayment time (days) for paid loans.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns "Funded Date", "Paid Date", and the chosen features.
        Only rows with Status == "paid" should be passed.
    features : list of str
        Feature column names to use for training.

    Returns
    -------
    dict
        Dictionary with results for each model, including fitted pipeline, accuracy metrics,
        and system metrics (train time, inference time per sample, model size).
    """

    # --- Target variable: repayment time ---
    df = df.copy()
    df["repayment_days"] = (df["Paid Date"] - df["Funded Date"]).dt.days
    df = df.dropna(subset=["repayment_days"])
    df = df[df["repayment_days"] > 0]

    X = df[features]
    y = df["repayment_days"]

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Preprocessing ---
    # Use robust dtype checks (works for ints/floats, incl. pandas nullable types)
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    categorical_features = [f for f in features if f not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  # keep as-is; change to with_mean=False only if you must preserve sparsity end-to-end
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # --- Models ---
    models = {
        "linreg": LinearRegression(),
        "xgb": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        ),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in tqdm(models.items(), desc="Training repayment time models"):
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # --- defaults to avoid "might be referenced before assignment"
        train_time_s = float("nan")
        infer_ms_per_sample = float("nan")
        model_size_mb = float("nan")

        # --- Train (timed)
        t0 = perf_counter()
        pipe.fit(X_train, y_train)
        train_time_s = perf_counter() - t0

        # --- Predict on test (timed)
        t0 = perf_counter()
        preds = pipe.predict(X_test)
        infer_ms_per_sample = (perf_counter() - t0) * 1000.0 / max(len(X_test), 1)

        # --- Accuracy metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # --- Model size (in MB)
        try:
            buf = io.BytesIO()
            joblib.dump(pipe, buf)
            model_size_mb = len(buf.getbuffer()) / (1024 * 1024)
        except Exception:
            # keep NaN on failure
            pass

        results[name] = {
            "model": pipe,
            "mae": mae,
            "rmse": rmse,
            "train_time_s": train_time_s,
            "infer_ms_per_sample": infer_ms_per_sample,
            "model_size_mb": model_size_mb,
        }

    return results

