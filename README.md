# CVD Prediction (PySpark)

Forecasts national average cardiovascular disease (CVD) death rates in the United States using county-level data. The pipeline aggregates county rates per year and trains a simple regression model to predict future years (e.g., 2023 and 2030).

## Project Structure
- `dataset/CVD.csv` — raw dataset (county-level data, 2010–2020).
- `main.py` — PySpark pipeline to aggregate, train, evaluate, and predict.
- `outputs/` — results saved here (created on first run).
- `CVD_Analysis.ipynb` — notebook with an exploratory version (optional).

## Setup
1. Python 3.9+ recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you prefer, you can install just PySpark:
   ```bash
   pip install pyspark pandas matplotlib
   ```

## Usage
Run the pipeline from the project root:
```bash
python main.py --data_path dataset/CVD.csv --predict_years 2023 2030 --out_dir outputs
```

### What it does
- Loads county-level CVD death rates (`Data_Value`) and `Year`.
- Cleans and aggregates to national yearly averages.
- Trains a linear regression on `Year -> death_rate`.
- Evaluates RMSE/R².
- Predicts for requested years and saves results.

### Outputs
Saved under `outputs/`:
- `national_yearly_rates/` (CSV): national average per year.
- `predictions/` (CSV): predicted death rates for requested years.
- `evaluation.txt`: model metrics (RMSE, R², coefficients).

## Notes
- The provided dataset is county-level; we aggregate to a national average. If you need state-level or stratified insights, adjust the aggregation in `main.py`.
- For very large datasets, consider Git LFS. By default `.gitignore` excludes raw CSVs to keep the repository lean; you can commit the dataset if desired.

## GitHub Ready
- Includes `requirements.txt` and `.gitignore` to minimize repo noise.
- Deterministic outputs for easy verification.