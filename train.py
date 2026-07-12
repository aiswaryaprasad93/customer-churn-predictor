import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['TenureBucket'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=[1,2,3,4])
df['TenureBucket'] = df['TenureBucket'].astype(float)
df.fillna