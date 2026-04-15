/*
File: 04_behavioral_patterns.sql
Purpose: Analyze behavioral patterns influencing churn and retention
Dataset: customer_analytics
*/

-- =========================================
-- Q1: Churn Rate by Purchase Frequency
-- =========================================
-- Business Question:
-- Do customers who order more frequently churn less?

SELECT 
    frequency,
    COUNT(*) AS customers,
    SUM(churned_proxy) AS churned,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct
FROM customer_analytics
GROUP BY frequency
ORDER BY frequency DESC;


-- =========================================
-- Q2: Churn vs Number of Unique Products
-- =========================================
-- Business Question:
-- Does buying more diverse products reduce churn?

SELECT 
    CASE 
        WHEN unique_products = 1 THEN '1 Product'
        WHEN unique_products BETWEEN 2 AND 5 THEN '2-5 Products'
        WHEN unique_products BETWEEN 6 AND 10 THEN '6-10 Products'
        WHEN unique_products BETWEEN 11 AND 20 THEN '11-20 Products'
        ELSE '20+ Products'
    END AS product_range,
    COUNT(*) AS customers,
    ROUND(AVG(churn_probability), 4) AS avg_churn_prob,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct
FROM customer_analytics
GROUP BY product_range
ORDER BY churn_rate_pct DESC;


-- =========================================
-- Q3: Avg Order Value vs Churn
-- =========================================
-- Business Question:
-- Do higher spending customers churn less?

SELECT 
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    churned_proxy,
    COUNT(*) AS customers
FROM customer_analytics
GROUP BY churned_proxy;


-- =========================================
-- Q4: Recency vs Churn Probability
-- =========================================
-- Business Question:
-- Does time since last purchase influence churn probability?

SELECT 
    recency_level,
    COUNT(*) AS customers,
    ROUND(AVG(churn_probability), 4) AS avg_churn_prob,
    ROUND(AVG(recency), 1) AS avg_days_since_last_purchase
FROM customer_analytics
GROUP BY recency_level
ORDER BY avg_days_since_last_purchase DESC;

-- =========================================
-- Q5: Champions with Highest Risk (Relative)
-- =========================================
-- Business Question:
-- Among top customers, who has relatively higher churn risk?

SELECT 
    customer_id,
    segment,
    total_revenue,
    churn_probability,
    recency,
    frequency
FROM customer_analytics
WHERE segment = 'Champions'
ORDER BY churn_probability DESC
LIMIT 20;


-- Insight:
-- Champions exhibit extremely low churn probability, indicating strong loyalty.
-- However, a small increase in churn risk among high-revenue customers can lead
-- to significant financial impact. Monitoring even low-risk Champions is critical
-- due to their disproportionately high contribution to revenue.

-- =========================================
-- Q6: One-Time vs Repeat Customers
-- =========================================
-- Business Question:
-- Are one-time buyers more likely to churn?

SELECT 
    CASE 
        WHEN total_orders = 1 THEN 'One-Time Buyers'
        ELSE 'Repeat Buyers'
    END AS customer_group,
    COUNT(*) AS customers,
    SUM(churned_proxy) AS churned,
    ROUND(100.0 * SUM(churned_proxy) / COUNT(*), 2) AS churn_rate_pct
FROM customer_analytics
GROUP BY customer_group;


-- =========================================
-- Q7: Tenure vs Churn Probability
-- =========================================
-- Business Question:
-- Does longer customer lifetime reduce churn risk?

SELECT 
    CASE 
        WHEN tenure_days < 100 THEN 'New'
        WHEN tenure_days BETWEEN 100 AND 300 THEN 'Growing'
        ELSE 'Long-Term'
    END AS tenure_group,
    COUNT(*) AS customers,
    ROUND(AVG(churn_probability), 4) AS avg_churn_prob,
    ROUND(AVG(total_revenue), 2) AS avg_revenue
FROM customer_analytics
GROUP BY tenure_group
ORDER BY avg_churn_prob DESC;

-- =========================================
-- Q8: Revenue concentration among top Champions
-- =========================================
-- Business Question:
-- How much revenue is concentrated among top Champions?

SELECT 
    SUM(total_revenue) AS top_champion_revenue
FROM (
    SELECT total_revenue
    FROM customer_analytics
    WHERE segment = 'Champions'
    ORDER BY total_revenue DESC
    LIMIT 10
) t;