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
            print(f"\tid: {sample['id']}, Funded Date: {sample['Funded Date']}, Paid Date: {sample['Paid Date']}, predicted_days: {sample['predicted_days']}, predicted_paid_date: {sample['predicted_paid_date']}")


if __name__ == "__main__":
    df = load_data()

    # Run both parts
    run_status_prediction(df)
    run_repayment_prediction(df)
