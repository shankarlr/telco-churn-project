# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed_with_features.csv")
MODEL_PATH = Path("models/simple_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Telco Churn — Simple Demo")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Make sure you've run the pipeline. Run: python run_all.py")
else:
    df = load_data()
    st.subheader("Dataset sample")
    st.dataframe(df.head())

    churn_rate = df['Churn'].map({'Yes':1,'No':0}).mean() if df['Churn'].dtype==object else df['Churn'].mean()
    st.metric("Churn rate", f"{churn_rate*100:.2f}%")

    st.markdown("---")
    st.subheader("Predict churn for a single customer")

    with st.form("single_pred"):
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)
        total = st.number_input("TotalCharges", min_value=0.0, value=800.0)
        num_services = st.number_input("Number of services", min_value=0, value=3)
        contract = st.selectbox("Contract", df['Contract'].unique())
        submitted = st.form_submit_button("Predict")

    if submitted:
        model = load_model()
        X = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "num_services": num_services,
            "Contract": contract
        }])
        proba = model.predict_proba(X)[:,1][0]
        st.write(f"Churn probability: {proba:.2%}")
        if proba > 0.5:
            st.warning("High churn risk — recommend retention action.")
        else:
            st.success("Low churn risk.")
