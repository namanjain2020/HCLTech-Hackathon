1. Project Overview

Retail businesses are not only interested in how much a customer has spent in the past, but more importantly in how much they are likely to spend in the near future.
This project aims to build a regression-based machine learning system that predicts customer purchase value over the next 30 days, also known as short-term Customer Lifetime Value (CLV).

2. Business Problem Statement

For each active customer, estimate:
“How much money will this customer spend in the next 30 days?”
Accurate short-term spend predictions help businesses:
Identify high-value customers
Optimize marketing and promotions
Allocate retention resources efficiently
Forecast short-term revenue

3. Machine Learning Problem Formulation
Aspect	Description
Learning Type	Supervised Learning
Task	Regression
Unit of Prediction	Customer
Input (X)	Aggregated historical customer behavior
Target (y)	Total spend in the next 30 days
Prediction Horizon	30 days
4. Dataset Selection
Chosen Dataset

UCI Online Retail II Dataset

The dataset satisfies all problem requirements:

Transaction-level retail data->Invoive No.
Unique customer identifiers->CustomerID
Timestamped purchase records->InvoiceDate
Monetary values computable per transaction->Quantity
Multiple purchases per customer->UnitPrice
Publicly available and widely used->Country

Transaction value is computed as:

TransactionAmount = Quantity × UnitPrice

5. Cut-Off Date Design (Data Leakage Prevention)

To ensure realistic and leakage-free modeling, we adopt a cut-off date based design.

Definition

A cut-off date separates:

Past behavior used to build features

Future behavior used to compute the target

Design Logic

Features (X): All transactions before the cut-off date

Target (y): Total spend in the 30 days after the cut-off date

Example
Cut-off Date: 2011-09-01

Features: InvoiceDate < 2011-09-01
Target:   Total spend from 2011-09-01 to 2011-10-01


This design ensures:

No future information leakage

Alignment with real-world deployment scenarios

6. Feature Engineering Plan (High-Level)

Features will be constructed at the customer level using historical data only.

Planned Feature Categories
6.1 RFM Features

Recency (days since last purchase)

Frequency (number of transactions)

Monetary value (total historical spend)

Average order value

6.2 Temporal Behavior

Spend in last 7 / 30 / 90 days

Purchase frequency per month

Spend trend ratios

6.3 Basket & Product Signals

Number of unique products purchased

Average quantity per order

Variability in order value

6.4 Geography (Optional)

Country encoding

7. High-Level System Architecture
Raw Transactions (CSV)
        |
        v
Data Cleaning & Validation
        |
        v
Cut-Off Date Logic
        |
        v
Customer-Level Feature Engineering
        |
        v
Regression Model
        |
        v
Saved Model & Preprocessing Pipeline
        |
        v
Inference API (Future Sprint)

8. Modeling Strategy (Planned)

Sprint-1 focuses on design only.
Planned models for future sprints include:

Baseline: Mean predictor, Linear Regression

Advanced: Gradient Boosting (LightGBM / XGBoost)

Evaluation Metrics: MAE, RMSE, R²

9. Technology Stack
Layer	Tools
Programming Language	Python
Data Processing	Pandas, NumPy
Modeling	Scikit-learn, LightGBM
Visualization	Matplotlib, Seaborn
Model Packaging	Joblib
API (Future)	FastAPI / Flask
10. Sprint-1 Scope & Deliverables
Sprint-1 Objective


11. Team Roles (If Applicable)
Role	Responsibility
Data Engineer	Data cleaning & preprocessing
ML Engineer	Feature design & modeling
Analyst	EDA & metric analysis
Backend Engineer	API & integration (future sprint)
12. Next Steps (Sprint-2 Preview)

Sprint-2 will focus on:

Data cleaning & EDA

Customer-level feature creation

Target variable construction

Baseline model training

13. Summary

Sprint-1 establishes a robust, leakage-free system design for predicting short-term customer spend.
The cut-off based formulation ensures realistic evaluation and smooth transition to production-ready modeling in subsequent sprints.


Write Sprint-2 README

Add architecture diagram image
