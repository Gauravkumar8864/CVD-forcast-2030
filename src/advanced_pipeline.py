import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def create_spark(app_name: str = "CVD_Advanced_Pipeline") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_and_clean(spark: SparkSession, csv_path: str) -> DataFrame:
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    # Keep columns we need; if extra columns exist it's ok
    df = df.withColumn("Year", col("Year").cast("int")).withColumn("Data_Value", col("Data_Value").cast("double"))
    # Filter reasonable years and non-null target
    df = df.filter((col("Year") >= 2010) & (col("Year") <= 2020) & col("Data_Value").isNotNull())
    return df


def aggregate_national(df: DataFrame) -> DataFrame:
    yearly = df.groupBy("Year").agg(avg("Data_Value").alias("death_rate")).orderBy("Year")
    return yearly


def train_spark_linear(yearly: DataFrame) -> Tuple[LinearRegression, VectorAssembler, DataFrame]:
    assembler = VectorAssembler(inputCols=["Year"], outputCol="features")
    features = assembler.transform(yearly).select("features", "death_rate")
    lr = LinearRegression(featuresCol="features", labelCol="death_rate")
    model = lr.fit(features)
    return model, assembler, features


def evaluate_spark_model(model, features_df: DataFrame) -> Dict:
    # model is a fitted spark regression model
    summary = model.summary
    # RMSE and R2 available from summary
    return {
        "rmse": float(summary.rootMeanSquaredError),
        "r2": float(summary.r2),
        "intercept": float(summary.intercept),
        "coefficients": [float(c) for c in summary.coefficients] if hasattr(summary, "coefficients") else []
    }


def train_spark_rf(yearly: DataFrame) -> Tuple[RandomForestRegressor, VectorAssembler, DataFrame]:
    assembler = VectorAssembler(inputCols=["Year"], outputCol="features")
    features = assembler.transform(yearly).select("features", "death_rate")
    rf = RandomForestRegressor(featuresCol="features", labelCol="death_rate", numTrees=200, maxDepth=8, seed=42)
    model = rf.fit(features)
    return model, assembler, features


def train_poly_via_numpy(yearly_pd: pd.DataFrame, degree: int = 2):
    # Fit polynomial on aggregated pandas series (Year vs death_rate)
    x = yearly_pd["Year"].values
    y = yearly_pd["death_rate"].values
    coeffs = np.polyfit(x, y, degree)
    return coeffs  # use np.polyval to predict


def predict_years_spark(model, assembler, years: List[int], spark: SparkSession) -> pd.DataFrame:
    # Create spark DF with future years and predict
    future = spark.createDataFrame([(int(y),) for y in years], ["Year"])
    future_features = assembler.transform(future).select("Year", "features")
    preds = model.transform(future_features).select("Year", col("prediction").alias("pred"))
    return preds.toPandas().sort_values("Year")


def compute_prediction_interval(preds_pd: pd.DataFrame, rmse: float) -> pd.DataFrame:
    # 95% interval approx by ±1.96*RMSE
    z = 1.96
    preds_pd = preds_pd.copy()
    preds_pd["lower"] = preds_pd["pred"] - z * rmse
    preds_pd["upper"] = preds_pd["pred"] + z * rmse
    return preds_pd


def plot_forecasts(history_pd: pd.DataFrame,
                   forecasts: Dict[str, pd.DataFrame],
                   out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history_pd["Year"], history_pd["death_rate"], "o-", label="Observed (2010-2020)", linewidth=2)

    # Colors and labeling
    for name, df in forecasts.items():
        if "pred" in df.columns:
            plt.plot(df["Year"], df["pred"], "--", marker="o", label=f"{name} prediction")
            # if intervals present, plot
            if "lower" in df.columns and "upper" in df.columns:
                plt.fill_between(df["Year"], df["lower"], df["upper"], alpha=0.2)
        else:
            plt.plot(df["Year"], df["Predicted_Mean_DeathRate"], "--", marker="o", label=f"{name} prediction")

    plt.xlabel("Year")
    plt.ylabel("Death rate")
    plt.title("CVD death rate: observed (2010-2020) and forecasts (2021-2030)")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(out_dir, "cvd_forecasts_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def state_level_forecast_2030(spark: SparkSession, df: DataFrame) -> pd.DataFrame:
    # For each state (LocationAbbr) compute linear trend via pandas and predict 2030 using degree=1 poly
    states = [r["LocationAbbr"] for r in df.select("LocationAbbr").distinct().collect()]
    rows = []
    for st in states:
        st_df = df.filter(df["LocationAbbr"] == st)
        if st_df.count() < 3:
            continue
        yearly = st_df.groupBy("Year").agg(avg("Data_Value").alias("rate")).orderBy("Year").toPandas()
        if len(yearly) < 2:
            continue
        # Fit deg1 polynomial for state
        coef = np.polyfit(yearly["Year"].values, yearly["rate"].values, 1)
        pred2030 = float(np.polyval(coef, 2030))
        rows.append({"LocationAbbr": st, "Year": 2030, "Predicted_Rate": pred2030})
    return pd.DataFrame(rows)


def run_advanced_pipeline(csv_path: str,
                          out_dir: str = "outputs",
                          predict_years: List[int] = list(range(2021, 2031))):
    spark = create_spark()
    try:
        print("Loading and cleaning data...")
        df = load_and_clean(spark, csv_path)

        print("Aggregating national yearly averages...")
        yearly = aggregate_national(df)  # spark DF Year, death_rate
        yearly_pd = yearly.toPandas().sort_values("Year")

        # Train linear model (Spark)
        print("Training linear regression (Spark)...")
        lr_model, lr_assembler, lr_features = train_spark_linear(yearly)
        lr_metrics = evaluate_spark_model(lr_model, lr_features)
        print("Linear model metrics:", lr_metrics)

        # Train Random Forest (Spark)
        print("Training Random Forest (Spark)...")
        rf_model, rf_assembler, rf_features = train_spark_rf(yearly)
        rf_metrics = evaluate_spark_model(rf_model, rf_features)
        print("Random Forest metrics:", rf_metrics)

        # Train polynomial (numpy) degree=2 on aggregated pandas series
        print("Fitting polynomial degree=2 on aggregated series...")
        poly_coeffs = train_poly_via_numpy(yearly_pd, degree=2)

        # Predictions for each model (2021-2030)
        print("Predicting years", predict_years)
        lr_preds = predict_years_spark(lr_model, lr_assembler, predict_years, spark)
        rf_preds = predict_years_spark(rf_model, rf_assembler, predict_years, spark)

        poly_preds = pd.DataFrame({
            "Year": predict_years,
            "pred": np.polyval(poly_coeffs, predict_years)
        })

        # Compute prediction intervals using RMSE from test (use metrics' rmse)
        lr_preds = compute_prediction_interval(lr_preds, lr_metrics["rmse"])
        rf_preds = compute_prediction_interval(rf_preds, rf_metrics["rmse"])

        # Save CSVs
        os.makedirs(out_dir, exist_ok=True)
        yearly_pd.to_csv(os.path.join(out_dir, "national_yearly_rates_2010_2020.csv"), index=False)
        lr_preds.to_csv(os.path.join(out_dir, "predictions_linear_2021_2030.csv"), index=False)
        rf_preds.to_csv(os.path.join(out_dir, "predictions_rf_2021_2030.csv"), index=False)
        poly_preds.to_csv(os.path.join(out_dir, "predictions_poly2_2021_2030.csv"), index=False)

        # Plotting
        plots_dir = os.path.join(out_dir, "plots")
        forecasts = {
            "Linear (Spark)": lr_preds,
            "RandomForest (Spark)": rf_preds,
            "Poly2 (aggregated)": poly_preds
        }
        plot_file = plot_forecasts(yearly_pd, forecasts, plots_dir)
        print("Saved forecast comparison plot to:", plot_file)

        # State-level 2030
        print("Computing state-level 2030 linear forecasts...")
        state_pred_df = state_level_forecast_2030(spark, df)
        state_pred_df.to_csv(os.path.join(out_dir, "state_level_2030_predictions.csv"), index=False)

        # Save summary evaluation
        eval_summary = {
            "linear": lr_metrics,
            "random_forest": rf_metrics,
            "poly2_coeffs": poly_coeffs.tolist()
        }
        with open(os.path.join(out_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
            f.write(str(eval_summary))

        print("Advanced pipeline complete. Outputs in:", out_dir)
        return {
            "out_dir": out_dir,
            "plots": plot_file,
            "lr_metrics": lr_metrics,
            "rf_metrics": rf_metrics,
            "poly_coeffs": poly_coeffs
        }
    finally:
        spark.stop()
