import os
import argparse
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


def create_spark(app_name: str = "CVD_Prediction") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_data(spark: SparkSession, csv_path: str) -> DataFrame:
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    return df


def clean_and_select(df: DataFrame) -> DataFrame:
    # Select needed columns and cast types
    # Data_Value is the CVD rate per 100,000 for ages 35-64, spatially smoothed
    cleaned = (
        df.select(
            col("Year").cast("int").alias("year"),
            col("Data_Value").cast("double").alias("rate")
        )
        .where(col("year").isNotNull() & col("rate").isNotNull())
    )
    return cleaned


def aggregate_national_yearly(df: DataFrame) -> DataFrame:
    # County-level -> national average per year
    yearly = (
        df.groupBy("year").agg(avg("rate").alias("death_rate"))
        .orderBy("year")
    )
    return yearly


def train_model(yearly: DataFrame):
    assembler = VectorAssembler(inputCols=["year"], outputCol="features")
    features_df = assembler.transform(yearly)

    lr = LinearRegression(featuresCol="features", labelCol="death_rate")
    model = lr.fit(features_df)
    return model, assembler, features_df


def evaluate_model(model: LinearRegression, features_df: DataFrame):
    summary = model.summary
    return {
        "rmse": float(summary.rootMeanSquaredError),
        "r2": float(summary.r2),
        "coefficients": [float(c) for c in summary.coefficients],
        "intercept": float(summary.intercept),
        "num_instances": int(summary.totalIterations) if hasattr(summary, "totalIterations") else None,
    }


def predict_years(spark: SparkSession, assembler: VectorAssembler, model: LinearRegression, years: List[int]) -> DataFrame:
    future_df = spark.createDataFrame([(int(y),) for y in years], ["year"])
    future_features = assembler.transform(future_df)
    preds = model.transform(future_features).select("year", col("prediction").alias("predicted_death_rate"))
    return preds.orderBy("year")


def save_outputs(yearly: DataFrame, eval_metrics: dict, preds: DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Save aggregated yearly national rates
    yearly.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_dir, "national_yearly_rates"))

    # Save predictions
    preds.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(out_dir, "predictions"))

    # Save evaluation metrics to a simple text file (for GitHub viewing)
    metrics_path = os.path.join(out_dir, "evaluation.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Model Evaluation\n")
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")


def main():
    parser = argparse.ArgumentParser(description="CVD death rate forecasting (national average) with PySpark")
    parser.add_argument("--data_path", type=str, default=os.path.join("dataset", "CVD.csv"), help="Path to CVD CSV dataset")
    parser.add_argument("--predict_years", type=int, nargs="*", default=[2023, 2030], help="Years to predict, e.g., --predict_years 2023 2030")
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs"), help="Directory to save outputs")
    args = parser.parse_args()

    spark = create_spark()
    try:
        df = load_data(spark, args.data_path)
        cleaned = clean_and_select(df)
        yearly = aggregate_national_yearly(cleaned)

        # Train and evaluate
        model, assembler, features_df = train_model(yearly)
        eval_metrics = evaluate_model(model, features_df)

        # Predict requested years
        preds = predict_years(spark, assembler, model, args.predict_years)

        # Save outputs
        save_outputs(yearly, eval_metrics, preds, args.out_dir)

        # Show quick console outputs
        print("\nNational yearly average rates (sample):")
        yearly.show(10)
        print("\nPredictions:")
        preds.show(len(args.predict_years))
        print("\nEvaluation metrics:")
        for k, v in eval_metrics.items():
            print(f"{k}: {v}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()