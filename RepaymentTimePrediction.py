import pandas as pd
import numpy as np
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
import math


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
        Dictionary with results for each model, including fitted pipeline and metrics.
    """

    # --- Target variable: repayment time ---
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
    numeric_features = [f for f in features if df[f].dtype in [np.int64, np.float64]]
    categorical_features = [f for f in features if f not in numeric_features]

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
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # --- Models ---
    models = {
        "linreg": LinearRegression(),
        "xgb": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }

    results = {}
    for name, model in tqdm(models.items(), desc="Training repayment time models"):
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {
            "model": pipe,
            "mae": mae,
            "rmse": rmse
        }

    return results
