# SHORT-TERM CUSTOMER LIFETIME VALUE (CLV) PREDICTION
---

## 1. PROJECT OVERVIEW

For retail businesses, understanding past customer spending is useful, but predicting future spending is far more valuable.
In this project, we develop a regression-based machine learning system to predict how much a customer is expected to spend in the next 30 days, also known as short-term Customer Lifetime Value (CLV).

---

## 2. BUSINESS PROBLEM STATEMENT

For each active customer, estimate:

How much money will this customer spend in the next 30 days?
Accurate short-term spend predictions enable businesses to:

- Identify high-value customers
- Optimize marketing and promotional strategies
- Allocate retention resources efficiently
- Forecast short-term revenue

---

## 3. MACHINE LEARNING PROBLEM FORMULATION

| Aspect | Description |
|------|------------|
| Learning Type | Supervised Learning |
| Task | Regression |
| Unit of Prediction | Done per Customer |
| Input (X) | Aggregated historical customer behavior |
| Target (y) | Total spend in the next 30 days |
| Prediction Horizon | 30 days |

## REGRESSION

- LINEAR REGRESSION
- SUPPORT VECTOR REGRESSION
- RANDOM FOREST REGRESSOR
- XG BOOST 

---

## 4. DATASET DESCRIPTION

### Dataset Used

UCI Online Retail II Dataset

### Dataset Properties

| Feature Description | Dataset Column / Detail |
|---------------------|------------------------|
| Transaction-level retail data | InvoiceNo |
| Unique customer identifiers | CustomerID |
| Timestamped purchase records | InvoiceDate |
| Monetary values per transaction | Quantity × UnitPrice |
| Multiple purchases per customer | Repeated InvoiceNo entries per CustomerID |
| Publicly available and widely used | UCI Online Retail II Dataset |


### Transaction Value Calculation

TransactionAmount = Quantity × UnitPrice

---

## 5. CUT-OFF DATE DESIGN 

To ensure realistic and leakage-free modeling, a cut-off date based design is used.
Cut-off Date: 01-10-2011
Features: InvoiceDate before 01-10-2011
Target:   Total spend from 01-10-2011 to 31-10-2011


### Concept

The cut-off date separates:

- Past customer behavior used for feature construction
- Future customer behavior used to compute the target

### Design Logic

- Features (X): All transactions before the cut-off date
- Target (y): Total spend in the 30 days after the cut-off date

---

## 6. FEATURE ENGINEERING STRATEGY

All features are computed at the customer level using historical data only.

### 6.1 RFM FEATURES

- Recency (days since last purchase)
- Frequency (number of transactions)
- Monetary value (total historical spend)
- Average order value

### 6.2 BASKET AND PRODUCT FEATURES

- Number of unique products purchased
- Average quantity per order
- Variability in order value

---

## 7. SYSTEM ARCHITECTURE
Libraries used:-

- Numpy
- Pandas
- MatPlotlib
- Seaborn
- Scikit learn
- tqdm
- Streamlit
- Joblib
- Fast API

### HIGH-LEVEL PIPELINE

Raw Transactions (CSV) -> Data Cleaning and Validation -> Cut-Off Date Logic -> Customer-Level Feature Engineering -> Regression Model -> Saved Model and Preprocessing Pipeline -> Streamlit (To display working)

### Rough Allocation of Roles 
- Gouri Patidar : Exploratory Data Analysis and feature engineering
- Naman Jain : Feature Engineering & Model Training
- Mohit Jingar : Inferential Pipeline
- Tanisha Mangliya : Deployment on Streamlit
