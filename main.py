import pandas as pd
from StatusPrediction import train_status_models
from RepaymentTimePrediction import train_repayment_time_model


def load_data(path="data/bigml.csv"):
    # Load CSV
    df = pd.read_csv(path)

    # Convert dates
    df["Funded Date"] = pd.to_datetime(df["Funded Date"], errors="coerce")
    df["Paid Date"] = pd.to_datetime(df["Paid Date"], errors="coerce")

    return df


def run_status_prediction(df):
    print("\n=== STATUS PREDICTION ===")

    # Train models (your existing function in StatusPrediction.py)
    results = train_status_models(df)

    # Print summary
    for model, res in results.items():
        print(f"\n[{model}]")
        print(f"AUC: {res['auc']:.4f}")
        print(f"Precision (defaulted): {res['precision0']:.3f} | Recall: {res['recall0']:.3f} | F1: {res['f1_0']:.3f}")


def run_repayment_prediction(df):
    print("\n=== REPAYMENT TIME PREDICTION ===")

    # Keep only paid loans
    paid_df = df[df["Status"] == "paid"].copy()

    # Choose features
    features = ["Funded Amount", "Loan Amount", "Country", "Sector"]

    results = train_repayment_time_model(paid_df, features)

    for model, res in results.items():
        print(f"\n[{model}]")
        print(f"MAE (days): {res['mae']:.2f}")
        print(f"RMSE (days): {res['rmse']:.2f}")

    # Example: predict repayment time for first 5 loans
    print("\nSample predictions:")

    for model, res in results.items():
        print(f"Model: {model}")
        samples = paid_df.head(5).copy()  # use 5 first rows to test prediction
        cur_model = results[model]["model"]
        samples["predicted_days"] = cur_model.predict(samples[features])
        samples["predicted_paid_date"] = samples["Funded Date"] + pd.to_timedelta(samples["predicted_days"], unit="D")
        for index, sample in samples.iterrows():
            print(
                f"\tid: {sample['id']}, Funded Date: {sample['Funded Date']}, Paid Date: {sample['Paid Date']}, predicted_days: {sample['predicted_days']}, predicted_paid_date: {sample['predicted_paid_date']}")


def print_system_summary(status_results, repay_results):
    if status_results:
        rows = []
        for name, r in status_results.items():
            rows.append({
                "Model": name,
                "Train time (s)": round(r.get("train_time_s", float("nan")), 3),
                "Infer time (ms/sample)": round(r.get("infer_ms_per_sample", float("nan")), 3),
                "Model size (MB)": round(r.get("model_size_mb", float("nan")), 3),
                "PR-AUC (defaulted)": round(r.get("pr_auc_defaulted", float("nan")), 4),
                "Brier (defaulted)": round(r.get("brier_defaulted", float("nan")), 4),
            })
        print("\n=== System & Prob. Metrics — Status Models ===")
        print(pd.DataFrame(rows).to_string(index=False))

    if repay_results:
        rows = []
        for name, r in repay_results.items():
            rows.append({
                "Model": name,
                "Train time (s)": round(r.get("train_time_s", float("nan")), 3),
                "Infer time (ms/sample)": round(r.get("infer_ms_per_sample", float("nan")), 3),
                "Model size (MB)": round(r.get("model_size_mb", float("nan")), 3),
                "MAE": round(r.get("mae", float("nan")), 3),
                "RMSE": round(r.get("rmse", float("nan")), 3),
            })
        print("\n=== System Metrics — Repayment Models ===")
        print(pd.DataFrame(rows).to_string(index=False))


def run_end_to_end_probabilistic(df, status_model_preference=("xgb", "rf", "logreg")):
    """
    1) Train status models and pick a model.
    2) Score P(paid) for ALL rows using the returned pipeline.
    3) Filter by the model's chosen threshold (picked on validation).
    4) Train repayment-time models on actually-paid rows.
    5) Predict repayment time ONLY for the filtered (predicted-paid) rows.
    """
    print("\n=== END-TO-END (probabilistic) ===")

    # ---- Stage 1: status (returns {model_name: {"model": pipe, "threshold": thr, ...}, ...})
    status_results = train_status_models(df)  # from StatusPrediction.py
    chosen_status = next((m for m in status_model_preference if m in status_results),
                         list(status_results.keys())[0])
    status_pipe = status_results[chosen_status]["model"]
    chosen_thr = status_results[chosen_status]["threshold"]
    print(f"[stage-1] chosen status model: {chosen_status}, threshold(P(paid))={chosen_thr:.3f}")

    # score probabilities for ALL rows with the status features
    STATUS_FEATURES = ["Country", "Loan Amount", "Sector", "Activity"]
    p_paid = status_pipe.predict_proba(df[STATUS_FEATURES])[:, 1]
    df_scored = df.copy()
    df_scored["p_paid"] = p_paid
    df_scored["pred_paid"] = (df_scored["p_paid"] >= chosen_thr).astype(int)

    eligible = df_scored[df_scored["pred_paid"] == 1].copy()
    print(f"[stage-1] eligible for stage-2 (predicted paid): {len(eligible)} / {len(df_scored)}")

    # ---- Stage 2: repayment-time (train on actually PAID)
    paid_df = df[df["Status"] == "paid"].copy()
    REPAY_FEATURES = ["Funded Amount", "Loan Amount", "Country", "Sector"]
    repay_results = train_repayment_time_model(paid_df, REPAY_FEATURES)  # from RepaymentTimePrediction.py

    # System metrics summary (training/inference time, model size, prob. metrics)
    print_system_summary(status_results, repay_results)

    chosen_repay = "xgb" if "xgb" in repay_results else next(iter(repay_results.keys()))
    repay_pipe = repay_results[chosen_repay]["model"]
    print(f"[stage-2] chosen repayment model: {chosen_repay}")

    # predict for eligible rows that have Funded Date (needed to compute a paid date)
    eligible = eligible.dropna(subset=["Funded Date"])
    if not eligible.empty:
        preds_days = repay_pipe.predict(eligible[REPAY_FEATURES])
        eligible["predicted_days"] = preds_days
        eligible["predicted_paid_date"] = eligible["Funded Date"] + pd.to_timedelta(eligible["predicted_days"],
                                                                                    unit="D")
        print("\n[stage-2] sample predictions on eligible rows:")
        print(eligible[["id", "Funded Date", "p_paid", "predicted_days", "predicted_paid_date"]].head(10).to_string(
            index=False))
    else:
        print("[stage-2] no eligible rows with valid 'Funded Date' to predict on.")

    return {
        "status_model": chosen_status,
        "status_threshold": chosen_thr,
        "repay_model": chosen_repay,
        "scored_df": df_scored,  # has p_paid + pred_paid
        "eligible_predictions": eligible  # only rows predicted-paid, with time predictions
    }


def print_versions():
    import sys, platform, numpy as np, pandas as pd, sklearn, xgboost
    print("\n=== Versions ===")
    print("Python:", sys.version.split()[0], "| OS:", platform.platform())
    print("numpy:", np.__version__, "pandas:", pd.__version__,
          "sklearn:", sklearn.__version__, "xgboost:", xgboost.__version__)


if __name__ == "__main__":
    print_versions()
    df = load_data()

    # Separate runs
    # run_status_prediction(df)
    # run_repayment_prediction(df)

    # end-to-end probabilistic pipeline
    _ = run_end_to_end_probabilistic(df)
