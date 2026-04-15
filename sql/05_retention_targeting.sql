/*
File: 05_retention_targeting.sql
Purpose: Identify high-priority customers for retention and estimate business impact
Dataset: customer_analytics
*/

-- =========================================
-- Q1: High-Risk Active Customers (Primary Target List)
-- =========================================
-- Business Question:
-- Which active customers are most likely to churn and should be targeted first?

SELECT 
    customer_id,
    country,
    segment,
    customer_type,
    total_revenue,
    churn_probability,
    revenue_at_risk
FROM customer_analytics
WHERE churned_proxy = 0
ORDER BY churn_probability DESC, revenue_at_risk DESC
LIMIT 20;


-- =========================================
-- Q2: Priority Score for Retention Targeting
-- =========================================
-- Business Question:
-- Can we rank customers using a combined risk + value score?

SELECT 
    customer_id,
    segment,
    total_revenue,
    churn_probability,
    revenue_at_risk,

    -- Priority Score Formula
    ROUND(
        (churn_probability * 0.5) +
        (revenue_at_risk / (SELECT MAX(revenue_at_risk) FROM customer_analytics) * 0.5),
        4
    ) AS priority_score

FROM customer_analytics
WHERE churned_proxy = 0
ORDER BY priority_score DESC
LIMIT 20;


-- =========================================
-- Q3: High-Value Customers at Risk
-- =========================================
-- Business Question:
-- Which high-value customers are at risk and need immediate attention?

SELECT 
    customer_id,
    segment,
    total_revenue,
    churn_probability,
    revenue_at_risk
FROM customer_analytics
WHERE customer_type = 'High Value'
  AND churned_proxy = 0
ORDER BY revenue_at_risk DESC
LIMIT 20;


-- =========================================
-- Q4: Segment-wise High-Risk Distribution
-- =========================================
-- Business Question:
-- Which segments contain the most high-risk customers?

SELECT 
    segment,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_customers,
    ROUND(100.0 * SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_high_risk
FROM customer_analytics
GROUP BY segment
ORDER BY pct_high_risk DESC;


-- =========================================
-- Q5: Medium-Risk Customers (Opportunity Zone)
-- =========================================
-- Business Question:
-- Which customers are not yet churned but show moderate risk?

SELECT 
    customer_id,
    segment,
    total_revenue,
    churn_probability,
    revenue_at_risk
FROM customer_analytics
WHERE ml_based_risk_segment = 'Medium Risk'
  AND churned_proxy = 0
ORDER BY churn_probability DESC
LIMIT 20;


-- =========================================
-- Q6: Estimated Retention ROI
-- =========================================
-- Business Question:
-- What is the estimated financial benefit of retention efforts?

SELECT 
    ml_based_risk_segment,
    COUNT(*) AS customers,
    SUM(revenue_at_risk) AS total_risk,

    -- Assume intervention success rate = 15%
    ROUND(SUM(revenue_at_risk) * 0.15, 2) AS estimated_revenue_saved

FROM customer_analytics
GROUP BY ml_based_risk_segment
ORDER BY total_risk DESC;


-- =========================================
-- Q7: Recommended Retention Strategy
-- =========================================
-- Business Question:
-- What type of intervention should be applied to different customer groups?

SELECT 
    customer_id,
    segment,
    ml_based_risk_segment,
    total_revenue,
    churn_probability,

    CASE 
        WHEN segment = 'Champions' AND churn_probability > 0.1 
            THEN 'VIP retention - personalized offer'
        WHEN ml_based_risk_segment = 'High Risk' 
            THEN 'Urgent intervention - discount or call'
        WHEN ml_based_risk_segment = 'Medium Risk' 
            THEN 'Engagement campaign - reminders'
        ELSE 'Low priority - monitor'
    END AS retention_strategy

FROM customer_analytics
WHERE churned_proxy = 0
ORDER BY churn_probability DESC
LIMIT 20;