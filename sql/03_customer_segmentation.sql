/*
File: 03_customer_segmentation.sql
Purpose: Analyze customer segments using RFM, ML risk, and behavioral metrics
Dataset: customer_analytics
*/

-- =========================================
-- Q1: Segment Distribution
-- =========================================
-- Business Question:
-- How are customers distributed across RFM segments?

SELECT 
    segment,
    COUNT(*) AS total_customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customer_analytics), 2) AS pct_of_total
FROM customer_analytics
GROUP BY segment
ORDER BY total_customers DESC;


-- =========================================
-- Q2: Segment Profile (Core Metrics)
-- =========================================
-- Business Question:
-- How do key metrics differ across segments?

SELECT 
    segment,
    COUNT(*) AS customers,
    ROUND(AVG(recency), 1) AS avg_recency,
    ROUND(AVG(frequency), 2) AS avg_frequency,
    ROUND(AVG(monetary), 2) AS avg_monetary,
    ROUND(AVG(clv_score), 2) AS avg_clv,
    ROUND(AVG(churn_probability), 4) AS avg_churn_prob
FROM customer_analytics
GROUP BY segment
ORDER BY avg_clv DESC;


-- =========================================
-- Q3: RFM vs ML Risk Comparison
-- =========================================
-- Business Question:
-- How do rule-based segments compare with ML-based risk predictions?

SELECT 
    segment,
    ml_based_risk_segment,
    COUNT(*) AS customers
FROM customer_analytics
GROUP BY segment, ml_based_risk_segment
ORDER BY segment, customers DESC;


-- =========================================
-- Q4: Segment-wise Revenue and Risk
-- =========================================
-- Business Question:
-- Which segments generate the most revenue and carry the highest risk?

SELECT 
    segment,
    COUNT(*) AS customers,
    SUM(total_revenue) AS total_revenue,
    SUM(revenue_at_risk) AS total_risk,
    ROUND(100.0 * SUM(revenue_at_risk) / SUM(total_revenue), 2) AS risk_percentage
FROM customer_analytics
GROUP BY segment
ORDER BY total_revenue DESC;


-- =========================================
-- Q5: Customer Type within Segments
-- =========================================
-- Business Question:
-- How are high-value vs low-value customers distributed across segments?

SELECT 
    segment,
    customer_type,
    COUNT(*) AS customers,
    ROUND(AVG(total_revenue), 2) AS avg_revenue
FROM customer_analytics
GROUP BY segment, customer_type
ORDER BY segment, customers DESC;


-- =========================================
-- Q6: Purchase Behavior by Segment
-- =========================================
-- Business Question:
-- How does purchase behavior vary across segments?

SELECT 
    segment,
    ROUND(AVG(total_orders), 2) AS avg_orders,
    ROUND(AVG(total_items), 2) AS avg_items,
    ROUND(AVG(unique_products), 2) AS avg_unique_products,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value
FROM customer_analytics
GROUP BY segment
ORDER BY avg_orders DESC;


-- =========================================
-- Q7: High-Risk Customers within Each Segment
-- =========================================
-- Business Question:
-- What proportion of each segment is classified as high risk?

SELECT 
    segment,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_customers,
    ROUND(100.0 * SUM(CASE WHEN ml_based_risk_segment = 'High Risk' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_high_risk
FROM customer_analytics
GROUP BY segment
ORDER BY pct_high_risk DESC;