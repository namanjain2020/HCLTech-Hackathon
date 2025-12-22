import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Retail CLV Predictor", layout="wide")

st.title("🛍️ Retail 30-Day CLV Prediction")

# ======================
# Sidebar input
# ======================
st.sidebar.header("Customer Inputs")

recency = st.sidebar.number_input("Recency (days)", 0, 365, 12)
frequency = st.sidebar.number_input("Frequency", 1, 500, 45)
total_quantity = st.sidebar.number_input("Total Quantity", 1.0, 10000.0, 380.0)
total_spend = st.sidebar.number_input("Total Spend", 0.0, 1_000_000.0, 12500.0)
avg_order_value = st.sidebar.number_input("Avg Order Value", 0.0, 100000.0, 278.0)
unique_products = st.sidebar.number_input("Unique Products", 1, 5000, 67)
country = st.sidebar.selectbox("Country", ["United Kingdom", "France", "Germany", "Netherlands"])

payload = {
    "recency": recency,
    "frequency": frequency,
    "total_quantity": total_quantity,
    "total_spend": total_spend,
    "avg_order_value": avg_order_value,
    "unique_products": unique_products,
    "country": country
}

# ======================
# Buttons
# ======================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔮 Predict CLV"):
        res = requests.post(f"{API_URL}/predict", json=payload)
        if res.status_code == 200:
            st.success(f"Predicted 30-Day CLV: ₹ {res.json()['predicted_30d_clv']}")
        else:
            st.error("Prediction failed")

with col2:
    if st.button("📊 View Model Metrics"):
        res = requests.get(f"{API_URL}/metrics")
        st.json(res.json())

with col3:
    if st.button("🧠 Model Info"):
        res = requests.get(f"{API_URL}/model_info")
        st.json(res.json())

# ======================
# Batch prediction
# ======================
st.divider()
st.subheader("📦 Batch CLV Prediction")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        records = df.to_dict(orient="records")
        res = requests.post(
            f"{API_URL}/batch_predict",
            json={"records": records}
        )

        preds = res.json()["predictions"]
        df["predicted_30d_clv"] = preds
        st.success("Batch prediction completed")
        st.dataframe(df)
