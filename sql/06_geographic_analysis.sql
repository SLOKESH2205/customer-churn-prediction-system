/*
File: 06_geographic_analysis.sql
Purpose: Analyze churn, revenue, and risk across geographic regions
Dataset: customer_analytics
*/

-- =========================================
-- Q1: Customers and Revenue by Country
-- =========================================
-- Business Question:
-- Which countries contribute the most customers and revenue?

SELECT 
    country,
    COUNT(*) AS total_customers,
    SUM(total_revenue) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_revenue_per_customer
FROM customer_analytics
GROUP BY country
HAVING COUNT(*) >= 10
ORDER BY total_revenue DESC;


-- =========================================
-- Q2: Churn Rate by Country
-- =========================================
-- Business Question:
-- Which countries have the highest churn rates?

SELECT 
    country,
    COUNT(*) AS customers,
    SUM(churned_proxy) AS churned,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct
FROM customer_analytics
GROUP BY country
HAVING COUNT(*) >= 10
ORDER BY churn_rate_pct DESC;


-- =========================================
-- Q3: Revenue at Risk by Country
-- =========================================
-- Business Question:
-- Which countries contribute the most to revenue at risk?

SELECT 
    country,
    COUNT(*) AS customers,
    SUM(revenue_at_risk) AS total_risk,
    ROUND(100.0 * SUM(revenue_at_risk) / (SELECT SUM(revenue_at_risk) FROM customer_analytics), 2) AS pct_of_total_risk
FROM customer_analytics
GROUP BY country
HAVING COUNT(*) >= 10
ORDER BY total_risk DESC;


-- =========================================
-- Q4: ML Risk Distribution by Country
-- =========================================
-- Business Question:
-- How does ML-based churn risk vary across countries?

SELECT 
    country,
    ml_based_risk_segment,
    COUNT(*) AS customers
FROM customer_analytics
GROUP BY country, ml_based_risk_segment
HAVING COUNT(*) >= 10
ORDER BY country, customers DESC;


-- =========================================
-- Q5: UK vs Non-UK Comparison
-- =========================================
-- Business Question:
-- How does the dominant market (UK) compare to others?

SELECT 
    CASE 
        WHEN country = 'United Kingdom' THEN 'UK'
        ELSE 'Non-UK'
    END AS region_group,
    COUNT(*) AS customers,
    SUM(total_revenue) AS total_revenue,
    SUM(revenue_at_risk) AS total_risk,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct
FROM customer_analytics
GROUP BY region_group;


-- =========================================
-- Q6: Top Countries by High-Risk Customers
-- =========================================
-- Business Question:
-- Which countries have the most high-risk customers?

SELECT 
    country,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_customers,
    ROUND(100.0 * SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_high_risk
FROM customer_analytics
GROUP BY country
HAVING COUNT(*) >= 10
ORDER BY pct_high_risk DESC;