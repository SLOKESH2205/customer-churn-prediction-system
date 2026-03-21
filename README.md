# Customer Churn Intelligence Hub

## Business Problem

Most churn projects stop at a probability score. That is not enough for a business team that needs to decide:

- which customers are actually at risk
- which segments are most exposed
- why those customers are churning
- what action should be taken first

This project upgrades a standard churn-classification app into a decision-support product that combines customer segmentation, churn prediction, explainability, and business recommendations in one workflow.

## Product Goal

Build a customer intelligence system that helps retention, marketing, and product teams answer:

- Which customer persona has the highest churn risk?
- Are we losing high-value customers or low-engagement customers?
- Which behavioral features are driving churn predictions?
- What action should each segment receive next?

## Solution Overview

The system combines unsupervised and supervised learning:

1. `KMeans` clustering groups customers into behavioral segments using frequency, value, tenure, and purchase cadence.
2. A churn model predicts customer churn probability using engineered customer-level features plus segment membership.
3. The Streamlit dashboard translates technical outputs into business personas, key insights, and recommended actions.

## Key Capabilities

### 1. Customer Segmentation

- Automatically generates customer personas from cluster-level feature averages
- Replaces generic labels like `Cluster 0` with business-ready names such as:
  - `High-value customers losing momentum`
  - `Growing repeat customers`
  - `Newly acquired light buyers`
- Displays segment size, risk level, and behavioral interpretation

### 2. Churn Prediction

- Scores each customer with churn probability
- Allows threshold-based churn labeling for business use
- Highlights highest-risk customers for retention teams

### 3. Churn by Segment

- Calculates churn rate by persona
- Shows where churn is concentrated across the portfolio
- Connects segmentation and classification into a single decision layer

### 4. Explainability

- Ranks top churn drivers using tree-model feature importance
- Generates business-readable explanations of why users churn
- Supports optional SHAP visualization in the dashboard

### 5. Business Recommendation Layer

For each customer persona, the app generates:

- retention strategy
- marketing action
- product improvement suggestion
- primary action
- targeting rule
- business owner

This makes the output directly useful for business teams instead of ending at model metrics.

### 6. What-If Impact Simulator

- Estimates how many customers could be saved under a chosen intervention uplift
- Calculates protected revenue, campaign cost, and net impact by segment
- Helps prioritize which personas are worth targeting first

### 7. Downloadable Decision Report

- Exports the full dashboard output as a portable report
- Includes personas, churn by segment, feature importance, insights, recommendations, and impact simulation

## Example Analyst-Style Insights

The dashboard automatically produces data-driven insights such as:

- Customers with larger order values but weaker purchase cadence show elevated churn, which suggests valuable accounts are disengaging between purchases.
- Newer customers churn earlier than the retained base, pointing to onboarding and habit-formation gaps.
- Lower engagement is strongly associated with churn in the highest-risk segment.
- The most exposed segment is commercially important because it also contributes above-average customer value.

## Business Impact

This system is designed to support real retention decisions:

- Prioritize intervention on high-value segments before revenue is lost
- Distinguish onboarding problems from long-term disengagement
- Give marketing a segment-specific reactivation plan
- Give product teams evidence on where adoption friction is hurting retention

## Dashboard Structure

The Streamlit app is organized into business-facing sections:

- `Data Overview`
- `Customer Segmentation (Personas)`
- `Churn Prediction`
- `Feature Importance / Explainability`
- `Churn by Segment`
- `Key Insights`
- `Business Recommendations`
- `What-If Impact Simulator`
- `Advanced Model Diagnostics`

## Project Structure

```text
ML-PROJECT-1/
├── app.py
├── artifacts/
├── data/
├── requirements.txt
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── modeling.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   └── components/
│       ├── feature_engineering.py
│       ├── model_trainer.py
│       └── segment_analytics.py
└── README.md
```

## Technical Design

### Preprocessing

- Transaction-level retail data is converted into customer-level features
- Feature engineering includes:
  - frequency
  - monetary value
  - tenure
  - average order value
  - unique items purchased
  - purchase rate
  - monetary per day

### Clustering

- KMeans is used to group customers by behavioral similarity
- Cluster outputs are summarized and translated into personas

### Modeling

- Tree-based churn modeling supports:
  - probability scoring
  - threshold-based labeling
  - feature importance ranking
  - customer-level inference

### Insight Generation

- Segment risk and churn concentration are analyzed automatically
- Insight rules convert model outputs into analyst-style conclusions
- Recommendation logic maps segment patterns to business actions

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
streamlit run app.py
```

### 3. Upload data

Upload a CSV or Excel file with transaction-level customer purchase data.

Required columns:

- `customer_id`
- `invoice`
- `invoicedate`
- `price`
- `quantity`

## Screenshots

Add screenshots here after running the dashboard:

- `[Placeholder] Data Overview`
- `[Placeholder] Customer Segmentation`
- `[Placeholder] Churn by Segment`
- `[Placeholder] Feature Importance`
- `[Placeholder] Business Recommendations`

## Why This Project Stands Out

This is not a notebook-style churn exercise. It behaves like a business intelligence product:

- it combines segmentation and churn into one workflow
- it generates interpretable personas instead of raw cluster IDs
- it surfaces analyst-style conclusions instead of generic observations
- it maps model outputs to concrete business actions

That makes it much closer to what a company would actually use for retention planning, CRM prioritization, and product decision support.
