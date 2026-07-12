# Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer will churn (leave the service).

**Live Demo:** [Click here to try the app](https://customer-churn-predictor-snvbxxsqivaukuk5zrjtuq.streamlit.app/)

## Problem Statement
Customer churn is a major problem for telecom companies. This app predicts which customers are likely to leave, allowing businesses to take proactive action.

## Dataset
- Source: Telco Customer Churn (Kaggle)
- 7043 customers, 21 features
- Target: Churn (Yes/No)

## Approach
- Data cleaning and feature engineering
- Compared Logistic Regression, Random Forest, and XGBoost
- Best model: Logistic Regression (AUC: 0.846, Accuracy: 80%)
- Deployed as interactive web app using Streamlit

## Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (deployment)
- SHAP (explainability)
- GitHub + Streamlit Cloud

## Results
| Model | AUC Score |
|-------|-----------|
| Logistic Regression | 0.846 |
| Random Forest | 0.824 |
| XGBoost | 0.819 |

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
