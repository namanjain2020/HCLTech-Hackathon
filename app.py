import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
country = st.sidebar.selectbox(
    "Country", ["United Kingdom", "France", "Germany", "Netherlands"]
)

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

# --------------------------------------------------
# 🔮 DO NOT CHANGE THIS (as requested)
# --------------------------------------------------
with col1:
    if st.button("🔮 Predict CLV"):
        res = requests.post(f"{API_URL}/predict", json=payload)
        if res.status_code == 200:
            pred = res.json()["predicted_30d_clv"]
            st.success(f"Predicted 30-Day CLV: ₹ {pred}")
        else:
            st.error("Prediction failed")




# --------------------------------------------------
# 🧠 Model Info + Feature Influence
# --------------------------------------------------
with col3:
    if st.button("🧠 Model Info"):
        try:
            res = requests.get(f"{API_URL}/model_info", timeout=5)

            if res.status_code != 200:
                st.error("Model info API failed")
            else:
                info = res.json()

                st.subheader("🧠 Model Overview")

                st.write(f"**Model Used:** {info.get('model_name', 'Unknown')}")
                st.write(f"**Target Variable:** {info.get('target', '30-day CLV')}")

                features = info.get("features", [])

                if features:
                    st.markdown("### 📌 Input Features")
                    st.write(", ".join(features))

                st.markdown("### 🔍 Business Interpretation")
                st.markdown("""
                - **Recency ↓** → More recent buyers spend more  
                - **Frequency ↑** → Loyal customers increase CLV  
                - **Total Spend ↑** → Strongest predictor  
                - **Country** → Captures regional behavior  
                """)

        except Exception as e:
            st.error(f"Error loading model info: {e}")


# ======================
# Sensitivity Visuals
# ======================
st.divider()
st.subheader("📈 CLV Sensitivity Analysis")

colA, colB = st.columns(2)

# -------- Recency vs CLV --------
with colA:
    st.markdown("### Recency vs Predicted 30-Day CLV")

    recency_range = np.linspace(0, 365, 25)
    clv_vals = []

    for r in recency_range:
        temp = payload.copy()
        temp["recency"] = int(r)
        res = requests.post(f"{API_URL}/predict", json=temp)
        clv_vals.append(res.json()["predicted_30d_clv"])

    fig, ax = plt.subplots()
    ax.plot(recency_range, clv_vals, marker="o")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Predicted CLV")
    ax.set_title("Effect of Recency on 30-Day CLV")
    st.pyplot(fig)

# -------- Frequency vs CLV --------
with colB:
    st.markdown("### Frequency vs Predicted 30-Day CLV")

    freq_range = np.linspace(1, 100, 25)
    clv_vals = []

    for f in freq_range:
        temp = payload.copy()
        temp["frequency"] = int(f)
        res = requests.post(f"{API_URL}/predict", json=temp)
        clv_vals.append(res.json()["predicted_30d_clv"])

    fig, ax = plt.subplots()
    ax.plot(freq_range, clv_vals, marker="o", color="green")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Predicted CLV")
    ax.set_title("Effect of Frequency on 30-Day CLV")
    st.pyplot(fig)

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
