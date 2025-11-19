# CVD Death Rate Forecasting (2010–2030) — PySpark + ML Pipeline

This project forecasts **U.S. Cardiovascular Disease (CVD) death rates** up to the year **2030** using **PySpark MLlib**, multi-model regression, and advanced time-trend analysis.

The goal was to build a **complete, production-grade forecasting system**, including:
- Full PySpark EDA  
- Clean data preprocessing  
- National trends (2010–2020)  
- Multi-model forecasting (Linear, Random Forest, Polynomial)  
- Prediction intervals  
- State-level forecasts for 2030  
- Plot visualizations  
- Modular & professional project structure  

This repository delivers all of that.

---

## 🚀 Features

### **✔ PySpark-Based EDA**
- Schema inspection  
- Missing value analysis  
- National yearly averages  
- Top states by mortality  
- Stratification analysis (age, gender, etc.)  
- Geographic distribution (longitude/latitude)  

### **✔ Advanced Forecasting Pipeline (2021–2030)**
Implemented in `src/advanced_pipeline.py`:
- Linear Regression (PySpark MLlib)  
- Random Forest Regression (PySpark MLlib)  
- Polynomial Regression (degree 2)  
- Forecast comparison plots  
- 95% prediction intervals  
- Automatic CSV outputs  

### **✔ State-Level Forecast (2030)**
Predicts CVD death rates for every U.S. state using Spark aggregations + trend modeling.

---

## 📂 Project Structure

CVD-DeathRate-Forecast/
│
├── dataset/
│ └── CVD.csv
│
├── notebooks/
│ └── CVD_Analysis.ipynb # Clean PySpark EDA notebook
│
├── src/
│ ├── init.py
│ ├── advanced_pipeline.py # Main forecasting pipeline
│
├── outputs/
│ ├── national_yearly_rates_2010_2020.csv
│ ├── predictions_linear_2021_2030.csv
│ ├── predictions_rf_2021_2030.csv
│ ├── predictions_poly2_2021_2030.csv
│ ├── state_level_2030_predictions.csv
│ ├── evaluation_summary.txt
│ └── plots/
│ └── cvd_forecasts_comparison.png
│
├── main.py
├── requirements.txt
└── .gitignore


---

## 🔧 Tech Stack

**Languages / Frameworks**
- Python 3.9+
- PySpark (MLlib)
- Pandas (for plotting only)
- Matplotlib & Seaborn

**ML Models**
- Linear Regression  
- Random Forest Regression  
- Polynomial Regression (degree 2)  

**Outputs**
- 2021–2030 national forecasts  
- 2030 state-level forecasts  
- Confidence intervals  
- Visual comparison plot  

---

## 📈 Forecasting Methodology

### **1. Data Preparation**
- Cast schema  
- Filter 2010–2020  
- Handle null values  
- Compute national & state-level yearly averages  

### **2. Model Training**
Models trained on aggregated 2010–2020 data:
- Linear Regression → baseline trend  
- Random Forest → non-linear patterns  
- Polynomial Regression → curved trend  

### **3. Forecasting (2021–2030)**
Each model outputs:
- Predicted CVD death rate  
- Lower/upper 95% prediction bounds  

### **4. Visualization**
Outputs stored under:


Includes:
- Observed trend (2010–2020)  
- Model forecasts (2021–2030)  
- Confidence intervals  
- Side-by-side comparison  

---

## ▶️ How to Run

### **Install dependencies**

### **Run the pipeline**

### **View results**
All outputs will appear in the `outputs/` folder.

---

## 📊 Results Summary

- Forecasted national CVD death rates (2021–2030)  
- Random Forest vs Linear vs Polynomial comparison  
- Prediction intervals for each year  
- Top states by mortality  
- State-level forecast for 2030  
- Evaluation metrics (RMSE, R², coefficients)  

---

## 🏆 Why This Project Stands Out

- Built fully with **PySpark** (rare + powerful skill)  
- High-quality, production-grade pipeline  
- Real forecasting problem with real-world relevance  
- Strong mix of engineering + data science  
- Clean folder structure, reproducible workflow  
- Perfect for resumes, interviews, and academic submission  

---

## 👨‍💻 Author

**Gaurav Sharma**  
Master of Science in Artificial Intelligence  
Focused on ML Engineering, PySpark, and AI-driven forecasting.

---

## 📬 Contact

For collaboration or discussion, feel free to reach out via GitHub or LinkedIn.

