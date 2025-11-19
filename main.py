
import argparse
from src.advanced_pipeline import run_advanced_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced CVD forecasting pipeline")
    parser.add_argument("--data_path", type=str, default="dataset/CVD.csv", help="Path to CVD CSV file")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to write outputs")
    parser.add_argument("--start", type=int, default=2021, help="Forecast start year (inclusive)")
    parser.add_argument("--end", type=int, default=2030, help="Forecast end year (inclusive)")
    return parser.parse_args()

def main():
    args = parse_args()
    years = list(range(args.start, args.end + 1))
    result = run_advanced_pipeline(csv_path=args.data_path, out_dir=args.out_dir, predict_years=years)
    print("Done. Summary:", result)

if __name__ == "__main__":
    main()

