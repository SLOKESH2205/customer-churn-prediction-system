/*
File: 01_churn_overview.sql
Purpose: Analyze overall churn and how it varies across customer segments and behavior
Dataset: customer_analytics
*/

-- =========================================
-- Q1: Overall churn rate
-- =========================================
-- Business Question:
-- What percentage of customers have churned?

SELECT 
    COUNT(*) AS total_customers,
    SUM(churned_proxy) AS total_churned,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(100.0 * (COUNT(*) - SUM(churned_proxy)) / COUNT(*), 2) AS retention_rate_pct
FROM customer_analytics;


-- =========================================
-- Q2: Churn rate by RFM Segment
-- =========================================
-- Business Question:
-- Which customer segments are most likely to churn?

SELECT 
    segment,
    COUNT(*) AS total_customers,
    SUM(churned_proxy) AS churned_customers,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(churn_probability), 4) AS avg_churn_probability
FROM customer_analytics
GROUP BY segment
ORDER BY churn_rate_pct DESC;


-- =========================================
-- Q3: Churn rate by Customer Type (High vs Low Value)
-- =========================================
-- Business Question:
-- Are high-value customers churning more or less than low-value customers?

SELECT 
    customer_type,
    COUNT(*) AS total_customers,
    SUM(churned_proxy) AS churned_customers,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(total_revenue), 2) AS avg_revenue_per_customer
FROM customer_analytics
GROUP BY customer_type
ORDER BY churn_rate_pct DESC;


-- =========================================
-- Q4: Churn rate by Recency Level
-- =========================================
-- Business Question:
-- Are customers who haven't purchased recently more likely to churn?

SELECT 
    recency_level,
    COUNT(*) AS total_customers,
    SUM(churned_proxy) AS churned_customers,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(recency), 1) AS avg_days_since_last_purchase
FROM customer_analytics
GROUP BY recency_level
ORDER BY avg_days_since_last_purchase DESC;


-- =========================================
-- Q5: ML vs Actual churn comparison
-- =========================================
-- Business Question:
-- How does predicted churn compare with actual churn behavior?

SELECT 
    ml_based_risk_segment,
    COUNT(*) AS total_customers,
    SUM(churned_proxy) AS actual_churned,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS actual_churn_rate,
    ROUND(AVG(churn_probability), 4) AS avg_predicted_probability
FROM customer_analytics
GROUP BY ml_based_risk_segment
ORDER BY actual_churn_rate DESC;


-- =========================================
-- Q6: High-risk overlap (Rule vs ML)
-- =========================================
-- Business Question:
-- How many customers are flagged high risk by BOTH rule-based and ML methods?

SELECT 
    COUNT(*) AS high_risk_overlap,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customer_analytics), 2) AS pct_of_total
FROM customer_analytics
WHERE rule_based_risk_tier = 'High Risk'
  AND ml_based_risk_segment = 'High Risk';
  

-- =========================================
-- Q7: Active customers vs churn risk
-- =========================================
-- Business Question:
-- Among active customers, how many are predicted high risk?

SELECT 
    COUNT(*) AS active_customers,
    SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_active,
    ROUND(100.0 * SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_high_risk
FROM customer_analytics
WHERE churned_proxy = 0;

SELECT 
    ml_based_risk_segment,
    churned_proxy,
    COUNT(*) AS count
FROM customer_analytics
GROUP BY ml_based_risk_segment, churned_proxy
ORDER BY ml_based_risk_segment, churned_proxy;


-- =========================================
-- Q8: Early warning using churn probability
-- =========================================
-- Business Question:
-- Can probability scores identify risky active customers?

SELECT 
    COUNT(*) AS active_customers,
    SUM(CASE WHEN churn_probability > 0.5 THEN 1 ELSE 0 END) AS high_risk_active,
    ROUND(100.0 * SUM(CASE WHEN churn_probability > 0.5 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_high_risk
FROM customer_analytics
WHERE churned_proxy = 0;

-- Insight:
-- Probability-based filtering helps identify at-risk customers earlier,
-- improving the model's usefulness for proactive retention strategies.
-- Insight:
-- The ML model shows perfect separation for High Risk and Low Risk segments,
-- but fails to identify active customers at high risk of churn.
-- This indicates the model is highly aligned with historical churn behavior
-- but lacks early predictive capability.

