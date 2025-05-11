# 🏡 House Price Prediction – End-to-End MLOps Project

This project focuses on predicting house prices using the powerful **XGBoost Regressor** algorithm. It follows a complete **MLOps pipeline**—from data ingestion to model deployment—ensuring reproducibility, scalability, and deployment readiness.

---

## 🎯 Objective

To build a high-performing machine learning model that estimates the **SalePrice** of residential properties based on features like square footage, quality, neighborhood, garage size, and more.

---

## 🗽 MLOps Pipeline Overview

![MLOps Pipeline](https://i.imgur.com/YasAOB0.png)

---

This pipeline includes the following stages:

1. **Input data** - Load the raw dataset from Kaggle.
2. **EDA** - Explore the dataset to understand structure, patterns, and correlations.
3. **Features selection** - Select important features based on EDA insights.
4. **Preprocess** - Handle missing values, encode categorical features, and scale numerical values.
5. **Train and evaluate the performance of multiple model** - Train various models including XGBoost Regressor and evaluate using R², MAE, RMSE.
6. **FastAPI** - Develop an API endpoint for serving the model.
7. **Weights & Biases** - Track experiments, visualize training metrics, and manage model versions.

---

## 📈 Dataset Overview

The dataset comes from [Kaggle’s House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It includes **79 features** describing houses and the **SalePrice** target.

**Sample Features**:

* `GrLivArea`: Above-ground living area (sqft)
* `GarageCars`: Garage size (in cars)
* `YearBuilt`: Year built
* `OverallQual`: Overall material/finish quality
* `Neighborhood`: House location group

---

## 🧰 Technologies Used

| Tool/Library        | Purpose                       |
| ------------------- | ----------------------------- |
| Python 3.x          | Programming language          |
| XGBoost             | Main regression model         |
| pandas, numpy       | Data processing               |
| matplotlib, seaborn | Visualization                 |
| scikit-learn        | Preprocessing & evaluation    |
| wandb               | Experiment tracking           |
| FastAPI             | Model deployment via REST API |
| GitHub Actions      | CI/CD automation              |
| Jupyter Notebook    | Exploratory development       |

---

## 📂 Project Structure

```
Mlop-prj/
│
├── 1_fetch_data.ipynb          # Load raw dataset
├── 2_eda.ipynb                 # Exploratory data analysis
├── 3_preprocessing.ipynb       # Feature engineering & cleaning
├── 4_train_test.ipynb          # Model training & validation
│
├── app/
│   └── main.py                 # FastAPI deployment script
│
├── .github/workflows/ci.yml    # GitHub Actions CI workflow
├── house-prices-advanced-regression-techniques.zip
├── requirements.txt
│
├── artifacts/                  # Trained model & outputs
└── wandb/                      # Weights & Biases logs
```

---
## 🧠 Model: XGBoost Regressor

We use **XGBoost**, a gradient boosting framework optimized for speed and performance. It is widely used in regression problems involving structured/tabular data.

### 🔍 Why XGBoost?

* 🚀 **Fast training** with parallel processing
* 🧠 **Boosting-based learning**: sequential correction of residuals
* ✅ **Handles missing values** internally
* 📉 **Regularization (L1 & L2)** to avoid overfitting
* 📊 **Feature importance visualization** for interpretability
* ⏹️ **Early stopping** to prevent unnecessary training

### 📊 Performance Metrics Used:

| Metric                             | Description                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| **R² Score**                       | Measures how well the model explains price variance (1 = perfect fit)             |
| **MAE (Mean Absolute Error)**      | Average absolute difference between predicted and actual prices                   |
| **RMSE (Root Mean Squared Error)** | Penalizes larger errors more heavily for better realism in high-price predictions |

These metrics are automatically logged in **Weights & Biases (wandb)**, allowing you to:

* Monitor training in real-time
* Compare runs with different hyperparameters
* Visualize performance trends across experiments

---

## 🔍 Experiment Tracking with Weights & Biases

This project integrates with **Weights & Biases (wandb)** to log:

* Training/validation metrics
* Hyperparameters
* Model artifacts
* Training history and comparison charts

### Setup

```bash
pip install wandb
wandb login
```

---

## 🚀 Model Deployment (FastAPI)

The trained XGBoost model is deployed using **FastAPI**, exposing a REST API for predictions.

### Key Features:

* ✅ Lightweight and fast
* 📦 Accepts JSON input and returns predicted price
* 🧪 Comes with interactive Swagger UI
* 📁 Ready for containerization (e.g., Docker)

---

## 📌 CI/CD with GitHub Actions

This project includes a GitHub Actions workflow for continuous integration. It automatically:

* Runs linting and tests on every push or pull request
* Ensures dependencies are installed and environment is consistent
* Enables easy scaling and deployment automation

---

## 📌 Future Improvements

* 🔧 Docker container for deployment
* 🧠 Model explanation (SHAP, LIME)
* 📆 Integration with cloud storage and ML model registry

---

## 🤝 Contributing

Contributions are welcome!

You can:

* Improve model performance
* Refactor code
* Write tests
* Add deployment enhancements
* Improve documentation

---

## ⭐ Support the Project

If you find this project useful, give it a ⭐ on GitHub!
It helps the project grow and reach more developers.
