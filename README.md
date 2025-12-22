# SHORT-TERM CUSTOMER LIFETIME VALUE (CLV) PREDICTION
---

## 1. PROJECT OVERVIEW

Retail businesses are not only interested in how much a customer has spent in the past, but more importantly in how much they are likely to spend in the near future.

This project builds a regression-based machine learning system to predict customer purchase value over the next 30 days, also known as short-term Customer Lifetime Value (CLV).

Sprint-2 focuses on implementing the data pipeline, feature engineering, target construction, and baseline modeling defined in Sprint-1.

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
| Unit of Prediction | Customer |
| Input (X) | Aggregated historical customer behavior |
| Target (y) | Total spend in the next 30 days |
| Prediction Horizon | 30 days |

---

## 4. DATASET DESCRIPTION

### Dataset Used

UCI Online Retail II Dataset

### Dataset Properties

- Transaction-level retail data
- Unique customer identifiers (CustomerID)
- Timestamped purchase records (InvoiceDate)
- Monetary values computable per transaction
- Multiple purchases per customer
- Publicly available and widely used dataset

### Transaction Value Calculation

TransactionAmount = Quantity × UnitPrice


---

## 5. CUT-OFF DATE DESIGN (DATA LEAKAGE PREVENTION)

To ensure realistic and leakage-free modeling, a cut-off date based design is used.

### Concept

The cut-off date separates:

- Past customer behavior used for feature construction
- Future customer behavior used to compute the target

### Design Logic

- Features (X): All transactions before the cut-off date
- Target (y): Total spend in the 30 days after the cut-off date

### Example

- Cut-off Date: 2011-09-01
- Feature Window: InvoiceDate < 2011-09-01
- Target Window: 2011-09-01 to 2011-10-01

### Benefits

- Prevents future information leakage
- Matches real-world deployment scenarios
- Enables reliable and fair evaluation

---

## 6. FEATURE ENGINEERING STRATEGY

All features are computed at the customer level using historical data only.

### 6.1 RFM FEATURES

- Recency (days since last purchase)
- Frequency (number of transactions)
- Monetary value (total historical spend)
- Average order value

### 6.2 TEMPORAL BEHAVIOR FEATURES

- Spend in the last 7, 30, and 90 days
- Purchase frequency per month
- Spend trend ratios

### 6.3 BASKET AND PRODUCT FEATURES

- Number of unique products purchased
- Average quantity per order
- Variability in order value

### 6.4 GEOGRAPHIC FEATURES (OPTIONAL)

- Country-level encoding

---

## 7. SYSTEM ARCHITECTURE

### HIGH-LEVEL PIPELINE

Raw Transactions (CSV)
|
v
Data Cleaning and Validation
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
Saved Model and Preprocessing Pipeline
|
v
Streamlit (To display working)
