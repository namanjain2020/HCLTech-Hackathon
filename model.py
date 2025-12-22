# ============================================================
# SHORT-TERM CUSTOMER LIFETIME VALUE (CLV) PREDICTION
# Train window : Before 01-10-2011
# Test window  : 01-10-2011 to 31-10-2011
# Dataset      : UCI Online Retail II (KaggleHub)
# Models       : LR | SVR | RF | XGBoost
# ============================================================


# =========================
# 1. IMPORTS
# =========================
import kagglehub
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8")


# =========================
# 2. DATA LOADING (KAGGLEHUB)
# =========================
path = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")

csv_files = []
for root, _, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Standardize column names
df.columns = df.columns.str.strip().str.replace(" ", "")
df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice"}, inplace=True)

print("Initial shape:", df.shape)


# =========================
# 3. DATA CLEANING
# =========================
df = df.dropna(subset=["CustomerID"])

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep valid transactions
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Transaction value
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("After cleaning:", df.shape)


# =========================
# 4. DATE DEFINITIONS
# =========================
TRAIN_END_DATE = datetime(2011, 10, 1)
TEST_START_DATE = datetime(2011, 10, 1)
TEST_END_DATE = datetime(2011, 10, 31)

train_txn = df[df["InvoiceDate"] < TRAIN_END_DATE]
test_txn = df[
    (df["InvoiceDate"] >= TEST_START_DATE) &
    (df["InvoiceDate"] <= TEST_END_DATE)
]


# =========================
# 5. TARGET (30-DAY FUTURE SPEND)
# =========================
future_spend = (
    test_txn.groupby("CustomerID")["TotalPrice"]
    .sum()
    .reset_index(name="target_30d_spend")
)


# =========================
# 6. FEATURE ENGINEERING (TRAIN ONLY)
# =========================
features = train_txn.groupby("CustomerID").agg(
    recency=("InvoiceDate", lambda x: (TRAIN_END_DATE - x.max()).days),
    frequency=("InvoiceNo", "nunique"),
    total_quantity=("Quantity", "sum"),
    total_spend=("TotalPrice", "sum"),
    avg_order_value=("TotalPrice", "mean"),
    unique_products=("StockCode", "nunique"),
    country=("Country", "first")
).reset_index()

data = features.merge(future_spend, on="CustomerID", how="left")
data["target_30d_spend"] = data["target_30d_spend"].fillna(0)


# =========================
# 7. BASELINE MODEL
# =========================
baseline_pred = np.full(len(data), data["target_30d_spend"].mean())
print("Baseline MAE:",
      mean_absolute_error(data["target_30d_spend"], baseline_pred))


# =========================
# 8. FINAL DATASET
# =========================
X = data.drop(columns=["CustomerID", "target_30d_spend"])
y = data["target_30d_spend"]

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = ["country"]

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)


# =========================
# 9. REGRESSION MODELS
# =========================
models = {
    "Linear Regression": LinearRegression(),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost Regressor": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
}


# =========================
# 10. TRAIN & EVALUATE
# =========================
results = []
trained_pipelines = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_pipelines[name] = pipeline

    print(f"\n{name}")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)


# =========================
# 11. FINAL COMPARISON
# =========================
results_df = pd.DataFrame(results).sort_values("MAE")
print("\nFINAL MODEL COMPARISON")
print(results_df)


# =========================
# 12. SAVE BEST MODEL
# =========================
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

joblib.dump(best_pipeline, "best_clv_model.pkl")
print(f"\nBest model saved: {best_model_name} → best_clv_model.pkl")


# =========================
# 13. SAVE FINAL DATASET
# =========================
data.to_csv("Clv_task.csv", index=False)
print("Customer-level dataset saved as Clv_task.csv")
