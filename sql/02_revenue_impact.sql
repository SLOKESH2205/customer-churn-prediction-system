/*
File: 02_revenue_impact.sql
Purpose: Analyze financial impact of churn and quantify revenue at risk
Dataset: customer_analytics
*/

-- =========================================
-- Q1: Total Revenue vs Revenue at Risk
-- =========================================
-- Business Question:
-- How much total revenue is generated, and how much of it is at risk?

SELECT 
    SUM(total_revenue) AS total_revenue,
    SUM(revenue_at_risk) AS total_revenue_at_risk,
    ROUND(100.0 * SUM(revenue_at_risk) / SUM(total_revenue), 2) AS pct_revenue_at_risk
FROM customer_analytics;


-- =========================================
-- Q2: Revenue at Risk by ML Risk Segment
-- =========================================
-- Business Question:
-- Which risk segment contributes most to revenue risk?

SELECT 
    ml_based_risk_segment,
    COUNT(*) AS customers,
    SUM(revenue_at_risk) AS total_risk,
    ROUND(AVG(revenue_at_risk), 2) AS avg_risk_per_customer
FROM customer_analytics
GROUP BY ml_based_risk_segment
ORDER BY total_risk DESC;


-- =========================================
-- Q3: Revenue Contribution by RFM Segment
-- =========================================
-- Business Question:
-- Which customer segments generate the most revenue?

SELECT 
    segment,
    COUNT(*) AS customers,
    SUM(total_revenue) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_revenue_per_customer
FROM customer_analytics
GROUP BY segment
ORDER BY total_revenue DESC;


-- =========================================
-- Q4: Revenue at Risk by RFM Segment
-- =========================================
-- Business Question:
-- Which segments contribute most to revenue at risk?

SELECT 
    segment,
    COUNT(*) AS customers,
    SUM(revenue_at_risk) AS total_risk,
    ROUND(100.0 * SUM(revenue_at_risk) / (SELECT SUM(revenue_at_risk) FROM customer_analytics), 2) AS pct_of_total_risk
FROM customer_analytics
GROUP BY segment
ORDER BY total_risk DESC;


-- =========================================
-- Q5: High-Value Customers vs Risk
-- =========================================
-- Business Question:
-- Are high-value customers contributing significantly to revenue at risk?

SELECT 
    customer_type,
    COUNT(*) AS customers,
    SUM(total_revenue) AS total_revenue,
    SUM(revenue_at_risk) AS total_risk,
    ROUND(100.0 * SUM(revenue_at_risk) / SUM(total_revenue), 2) AS risk_percentage
FROM customer_analytics
GROUP BY customer_type
ORDER BY total_risk DESC;


-- =========================================
-- Q6: Churned vs Active Revenue Comparison
-- =========================================
-- Business Question:
-- How does revenue differ between churned and retained customers?

SELECT 
    churned_proxy,
    COUNT(*) AS customers,
    SUM(total_revenue) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_revenue,
    SUM(revenue_at_risk) AS total_risk
FROM customer_analytics
GROUP BY churned_proxy;


-- =========================================
-- Q7: Top 10 Customers by Revenue at Risk
-- =========================================
-- Business Question:
-- Which individual customers represent the highest financial risk?

SELECT 
    customer_id,
    country,
    segment,
    total_revenue,
    revenue_at_risk,
    churn_probability
FROM customer_analytics
ORDER BY revenue_at_risk DESC
LIMIT 10;