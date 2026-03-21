# 🚀 Customer Churn Intelligence System

## TL;DR

An end-to-end customer intelligence system that combines segmentation, churn prediction, explainability, and business decision support to help organizations identify high-risk customers and take targeted retention actions.

---

## 🌐 Live Demo

👉 https://churn-intelligence-dashboard.streamlit.app

---

## 📊 Project Scale

* Processed **500K+ transactions**
* Generated insights for **4K+ customers**
* Built a full pipeline from **raw data → business decisions**

---

## 🎯 Business Problem

Most churn projects stop at predicting:

> *“Who will churn?”*

That is not enough.

Businesses need to know:

* Which customers actually matter?
* Which segments are at risk?
* Why are they churning?
* What action should be taken first?

---

## 💡 Solution Overview

This project builds a **Customer Churn Intelligence System** that combines:

* Customer Segmentation (Clustering)
* Churn Prediction (Classification)
* Explainability
* Business Recommendations
* Impact Simulation

👉 Turning raw data into **actionable business decisions**

---

## 🧠 Key Capabilities

### 1. Customer Segmentation (Personas)

* Groups customers using behavioral features:

  * Frequency
  * Monetary value
  * Tenure
  * Purchase patterns
* Converts clusters into business personas:

  * High-value customers losing momentum
  * Growing repeat customers
  * Newly acquired light buyers

---

### 2. Churn Prediction

* Predicts churn probability per customer
* Supports threshold-based churn labeling
* Identifies high-risk segments

---

### 3. Churn by Segment

* Measures churn rate across personas
* Identifies where churn is concentrated
* Combines segmentation + prediction into one decision layer

---

### 4. Explainability

* Feature importance analysis
* Identifies key churn drivers:

  * Tenure
  * Engagement (frequency)
  * Purchase behavior

---

### 5. Business Recommendation Layer

For each persona, the system generates:

* Retention strategy
* Marketing actions
* Product improvements
* Targeting rules
* Business ownership

👉 Output is directly usable by business teams

---

### 6. What-If Impact Simulator (🔥 Key Feature)

* Estimates:

  * Customers saved
  * Revenue protected
  * Campaign cost
  * Net impact
* Helps prioritize high-value segments

---

### 7. Downloadable Decision Report

Exports:

* Personas
* Churn insights
* Feature importance
* Recommendations
* Impact simulation

---

## 📈 Model Performance

### 🔹 Clustering (KMeans)

* **Silhouette Score:** 0.9710
* **Davies-Bouldin Index:** 0.3771

👉 Indicates strong separation between customer segments

---

### 🔹 Churn Prediction (Random Forest)

* **Accuracy:** 0.7008
* **Precision:** 0.6979
* **Recall:** 0.6759
* **F1 Score:** 0.6867
* **ROC-AUC:** 0.7622

👉 Model balances precision and recall while maintaining strong discriminative power

---

### 🔹 Confusion Matrix

```
[[415 158]
 [175 365]]
```

👉 Shows balanced performance across churn and non-churn classes

---

### 🔹 Detailed Benchmark Results

### 🔹 Clustering (Customer Segmentation)

| Model | Silhouette Score |
|------|----------------|
| KMeans | **0.5209** |
| MiniBatch KMeans | 0.5207 |
| Gaussian Mixture Model (GMM) | 0.3944 |

👉 KMeans was selected as the final clustering model due to the highest silhouette score and better cluster stability.

**Key Observations:**
- Identified 3 distinct customer segments
- One cluster represents a very small segment (~44 customers in test set), indicating niche or outlier behavior
- Remaining clusters represent dominant behavioral groups

---

### 🔹 Churn Prediction (Best Model: XGBoost)

| Metric | Score |
|------|------|
| Accuracy | **0.7381** |
| Precision | **0.7103** |
| Recall | **0.7889** |
| F1 Score | **0.7475** |
| ROC-AUC | **0.8164** |

👉 Model is optimized for **recall**, ensuring high detection of churn-prone customers.

---

### 🔹 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|----------|----------|--------|----------|--------|
| Logistic Regression | 0.7185 | 0.6803 | 0.8062 | 0.7379 | 0.8030 |
| Random Forest | 0.7245 | 0.6879 | 0.8045 | 0.7416 | 0.8062 |
| XGBoost | **0.7381** | **0.7103** | 0.7889 | **0.7475** | **0.8164** |

👉 XGBoost selected due to:
- Highest F1 score (best balance)
- Strong ROC-AUC (discriminative power)
- Stable performance across classes

---

### 🔹 Confusion Matrix (XGBoost)

```
[[412 186]
 [122 456]]
```

👉 Interpretation:
- Model correctly identifies majority of churn customers (high recall)
- Some false positives exist, acceptable for retention-focused systems

---

## 🖥️ Dashboard Features

* Interactive controls (threshold, cost, uplift)
* Persona-based insights
* Churn segmentation
* Business recommendations
* Impact simulation

---

## 🧱 Architecture

```
Raw Data → Feature Engineering → Clustering → Churn Model → Insights → Dashboard
```

---

## ⚙️ Technical Design

### Preprocessing

* Converts transaction data → customer-level features:

  * frequency
  * monetary value
  * tenure
  * avg order value
  * purchase rate

---

### Clustering

* KMeans-based segmentation
* Behavioral grouping of customers

---

### Modeling

* Random Forest classifier
* Probability-based churn prediction
* Feature importance extraction

---

### Insight Engine

* Segment-level risk analysis
* Automated insight generation
* Rule-based recommendation mapping

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost (optional experimentation)
* SHAP
* Streamlit
* Matplotlib, Seaborn

---

## ▶️ How to Run

```bash
git clone https://github.com/SLOKESH2205/customer-churn-prediction-system.git
cd customer-churn-prediction-system

pip install -r requirements.txt
python src/components/run_trainer.py
streamlit run app.py
```

---

## 📂 Project Structure

```
customer-churn-prediction-system/
├── app.py
├── requirements.txt
├── runtime.txt
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
```

---

## 💡 Example Insights

* High-value customers show increased churn when engagement drops
* Customers with large order value but low purchase frequency are at highest risk
* New users churn early due to weak onboarding and habit formation

---

## 🎯 Business Impact

* Prioritizes high-value customers before revenue loss
* Distinguishes onboarding issues vs long-term disengagement
* Enables targeted retention campaigns
* Provides product teams with actionable insights

---

## 🚀 Why This Project Stands Out

This is not a basic ML project.

It:

* Combines segmentation + prediction in one system
* Translates clusters into business personas
* Converts model outputs into decisions
* Simulates real-world business impact

👉 Designed as a **decision intelligence product**, not just a model

---

## 👨‍💻 Author

**Lokesh S**

---

⭐ If you found this useful, consider starring the repo!
