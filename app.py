import streamlit as st
import pandas as pd
import joblib

model    = joblib.load("churn_model.pkl")
scaler   = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.markdown("Enter customer details to predict if they will leave.")

col1, col2 = st.columns(2)

with col1:
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges   = monthly_charges * tenure

with col2:
    senior    = st.checkbox("Senior Citizen")
    paperless = st.checkbox("Paperless Billing", value=True)

if st.button("🔮 Predict Churn", type="primary"):
    sample = pd.DataFrame(columns=features)
    sample.loc[0] = 0
    sample['tenure']           = tenure
    sample['MonthlyCharges']   = monthly_charges
    sample['TotalCharges']     = total_charges
    sample['AvgMonthlySpend']  = monthly_charges
    sample['SeniorCitizen']    = int(senior)
    sample['PaperlessBilling'] = int(paperless)

    scaled = scaler.transform(sample)
    prob   = model.predict_proba(scaled)[0][1]
    label  = "🔴 Likely to Churn" if prob > 0.5 else "🟢 Likely to Stay"

    st.markdown("---")
    st.subheader(f"Prediction: {label}")
    st.metric("Churn Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    if prob > 0.5:
        st.warning("⚠️ Recommend: offer discount or contract upgrade.")
    else:
        st.success("✅ Customer appears stable.")
