import math
from typing import Dict, List

import pandas as pd


SEGMENT_FEATURES = [
    "frequency_log",
    "monetary_log",
    "tenure",
    "avg_order_value",
    "purchase_rate",
    "unique_items_purchased",
    "monetary_per_day",
]


def _available_features(df: pd.DataFrame) -> List[str]:
    return [col for col in SEGMENT_FEATURES if col in df.columns]


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline is None or pd.isna(baseline) or math.isclose(float(baseline), 0.0):
        return 1.0
    return float(value) / float(baseline)


def _classify_level(value: float, baseline: float, high: float = 1.15, low: float = 0.85) -> str:
    ratio = _safe_ratio(value, baseline)
    if ratio >= high:
        return "high"
    if ratio <= low:
        return "low"
    return "mid"


def _risk_from_churn(churn_rate_pct: float) -> str:
    if churn_rate_pct >= 40:
        return "High"
    if churn_rate_pct >= 15:
        return "Medium"
    return "Low"


def _risk_badge(risk_level: str) -> str:
    badges = {
        "High": "Red",
        "Medium": "Amber",
        "Low": "Green",
    }
    return badges.get(risk_level, risk_level)


def _feature_highlights(row: pd.Series, overall: pd.Series) -> List[str]:
    highlights = []
    mapping = {
        "frequency_log": "engagement frequency",
        "purchase_rate": "purchase cadence",
        "monetary_log": "customer value",
        "avg_order_value": "basket size",
        "tenure": "customer tenure",
        "unique_items_purchased": "product breadth",
        "monetary_per_day": "daily revenue contribution",
    }

    for feature, label in mapping.items():
        if feature not in row.index or feature not in overall.index:
            continue
        level = _classify_level(row[feature], overall[feature])
        if level == "high":
            highlights.append(f"Above-average {label}")
        elif level == "low":
            highlights.append(f"Below-average {label}")

    if not highlights:
        highlights.append("Performance is close to the portfolio average")

    return highlights[:4]


def _persona_name(row: pd.Series, overall: pd.Series) -> str:
    churn = row["predicted_churn_rate_pct"]
    engagement_level = _classify_level(row.get("purchase_rate", row.get("frequency_log", 0)), overall.get("purchase_rate", overall.get("frequency_log", 0)))
    value_level = _classify_level(row.get("monetary_log", 0), overall.get("monetary_log", 0))
    tenure_level = _classify_level(row.get("tenure", 0), overall.get("tenure", 0))
    frequency_level = _classify_level(row.get("frequency_log", 0), overall.get("frequency_log", 0))

    if churn >= 55 and value_level == "high":
        return "High-value customers losing momentum"
    if churn >= 55 and tenure_level == "low":
        return "New customers with early churn risk"
    if churn >= 45 and engagement_level == "low":
        return "Low-engagement customers at risk"
    if tenure_level == "low" and value_level == "low":
        return "Newly acquired light buyers"
    if value_level == "high" and engagement_level == "high" and churn < 35:
        return "Loyal high-value regulars"
    if tenure_level == "high" and churn < 35:
        return "Stable long-tenure customers"
    if frequency_level in {"mid", "high"} and churn < 35:
        return "Growing repeat customers"
    if value_level == "low" and engagement_level == "low":
        return "Low-value occasional buyers"
    return "Core customers with mixed signals"


def _behavior_interpretation(row: pd.Series, overall: pd.Series) -> str:
    engagement_level = _classify_level(row.get("purchase_rate", row.get("frequency_log", 0)), overall.get("purchase_rate", overall.get("frequency_log", 0)))
    value_level = _classify_level(row.get("monetary_log", 0), overall.get("monetary_log", 0))
    tenure_level = _classify_level(row.get("tenure", 0), overall.get("tenure", 0))
    churn_rate = row["predicted_churn_rate_pct"]

    if churn_rate >= 55 and value_level == "high":
        return "This segment still contributes meaningful revenue, but its churn risk suggests weakening stickiness and an urgent retention gap."
    if churn_rate >= 45 and engagement_level == "low":
        return "Customers in this segment are showing low repeat behavior, making disengagement the clearest precursor to churn."
    if tenure_level == "low" and churn_rate >= 40:
        return "These are relatively new customers who are not reaching healthy adoption quickly enough after onboarding."
    if tenure_level == "low" and value_level == "low":
        return "This segment is made up of newly acquired customers with small baskets and limited purchase history, so the focus should be on activation rather than rescue."
    if value_level == "high" and engagement_level == "high":
        return "This is the healthiest segment: customers buy often, spend more, and show the strongest relationship depth."
    if tenure_level == "high" and churn_rate < 35:
        return "Long customer history and lower churn indicate this segment is stable and can support expansion plays."
    if churn_rate < 35 and engagement_level in {"mid", "high"}:
        return "Customers in this segment are building repeat behavior and look like the best candidates for nurture and upsell programs."
    return "This segment sits near the middle of the portfolio and needs targeted nudges rather than aggressive intervention."


def _recommended_actions(row: pd.Series, overall: pd.Series) -> Dict[str, str]:
    value_level = _classify_level(row.get("monetary_log", 0), overall.get("monetary_log", 0))
    engagement_level = _classify_level(row.get("purchase_rate", row.get("frequency_log", 0)), overall.get("purchase_rate", overall.get("frequency_log", 0)))
    tenure_level = _classify_level(row.get("tenure", 0), overall.get("tenure", 0))
    churn_rate = row["predicted_churn_rate_pct"]

    if churn_rate >= 55 and value_level == "high":
        return {
            "primary_action": "VIP rescue campaign",
            "targeting_rule": "Prioritize customers above segment-average spend with churn probability above threshold.",
            "owner": "Retention + account management",
            "retention_strategy": "Launch concierge-style save offers for the top accounts in this segment before the next expected purchase window.",
            "marketing_action": "Run a high-touch win-back campaign with personalized incentives tied to prior product preferences.",
            "product_improvement": "Interview a sample of churn-prone high-value customers to identify friction in pricing, service, or product fit.",
        }
    if churn_rate >= 45 and engagement_level == "low":
        return {
            "primary_action": "Behavioral reactivation",
            "targeting_rule": "Target customers whose purchase cadence has dropped below the segment baseline.",
            "owner": "CRM marketing",
            "retention_strategy": "Trigger reactivation journeys when purchase cadence drops below the segment baseline.",
            "marketing_action": "Send behavior-based reminders, replenishment nudges, and low-friction return offers.",
            "product_improvement": "Simplify the repeat-purchase journey and surface the most relevant products earlier.",
        }
    if tenure_level == "low" and churn_rate >= 40:
        return {
            "primary_action": "Onboarding recovery",
            "targeting_rule": "Target new customers with weak second-purchase behavior in the first 30-45 days.",
            "owner": "Lifecycle marketing + product",
            "retention_strategy": "Strengthen first-30-day onboarding with milestone-based outreach and success checkpoints.",
            "marketing_action": "Promote starter bundles and educational campaigns that accelerate time to second purchase.",
            "product_improvement": "Improve activation flows so new customers discover value faster and with fewer steps.",
        }
    if tenure_level == "low" and value_level == "low":
        return {
            "primary_action": "Second-order activation",
            "targeting_rule": "Target first-time buyers who have not returned within the expected repeat window.",
            "owner": "Lifecycle marketing",
            "retention_strategy": "Focus on activation goals such as second-order conversion instead of using expensive save offers.",
            "marketing_action": "Run welcome journeys, starter-pack promotions, and habit-building reminders after the first purchase.",
            "product_improvement": "Reduce friction for repeat purchase and help new buyers find relevant products faster.",
        }
    if value_level == "high" and engagement_level == "high":
        return {
            "primary_action": "Protect and expand",
            "targeting_rule": "Target top-value repeat buyers for loyalty perks and cross-sell journeys.",
            "owner": "CRM marketing + loyalty team",
            "retention_strategy": "Protect this segment with loyalty recognition and proactive account care rather than discounts.",
            "marketing_action": "Use VIP cross-sell and referral campaigns to expand wallet share.",
            "product_improvement": "Invest in premium-tier experiences and exclusivity features that reward heavy usage.",
        }
    return {
        "primary_action": "Light-touch nurture",
        "targeting_rule": "Target customers drifting toward lower engagement before they enter the high-risk band.",
        "owner": "Growth marketing",
        "retention_strategy": "Use light-touch retention nudges and monitor for movement into higher-risk behavior bands.",
        "marketing_action": "Test segmented campaigns to increase order frequency without over-discounting.",
        "product_improvement": "Prioritize usability improvements that reduce friction in repeat purchasing and discovery.",
    }


def _portfolio_insights(summary_df: pd.DataFrame, overall: pd.Series) -> List[str]:
    if summary_df.empty:
        return []

    ranked = summary_df.sort_values("predicted_churn_rate_pct", ascending=False).reset_index(drop=True)
    highest = ranked.iloc[0]
    lowest = ranked.iloc[-1]
    insights = [
        f"{highest['persona_name']} is the highest-risk segment at {highest['predicted_churn_rate_pct']:.1f}% predicted churn, versus {lowest['predicted_churn_rate_pct']:.1f}% for {lowest['persona_name']}.",
    ]

    if "purchase_rate" in summary_df.columns and "tenure" in summary_df.columns:
        high_engagement = _classify_level(highest["purchase_rate"], overall["purchase_rate"])
        low_engagement = _classify_level(lowest["purchase_rate"], overall["purchase_rate"])
        if high_engagement == "low" and low_engagement != "low":
            insights.append(
                f"Lower engagement is strongly associated with churn here: {highest['persona_name']} buys less frequently than average while carrying the greatest risk."
            )
        if highest["purchase_rate"] < overall["purchase_rate"] and highest["tenure"] < overall["tenure"]:
            insights.append(
                f"{highest['persona_name']} combines below-average tenure with below-average engagement, which points to an onboarding-to-habit-formation gap rather than a pricing-only issue."
            )
        if highest["tenure"] < lowest["tenure"]:
            insights.append(
                f"Tenure appears protective in this portfolio: the riskiest segment averages {highest['tenure']:.1f} tenure days compared with {lowest['tenure']:.1f} days for the lowest-risk segment."
            )

    if "monetary_log" in summary_df.columns and _classify_level(highest["monetary_log"], overall["monetary_log"]) == "high":
        insights.append(
            f"The main churn exposure is commercially important because {highest['persona_name']} also sits above average on customer value."
        )
    if "avg_order_value" in summary_df.columns and "purchase_rate" in summary_df.columns:
        if highest["avg_order_value"] > overall.get("avg_order_value", highest["avg_order_value"]) and highest["purchase_rate"] < overall.get("purchase_rate", highest["purchase_rate"]):
            insights.append(
                f"{highest['persona_name']} tends to place larger orders but returns less often, suggesting high-value accounts are disengaging between purchases instead of fully downgrading spend."
            )

    return insights[:4]


def analyze_segments(result_df: pd.DataFrame) -> Dict[str, object]:
    if "cluster" not in result_df.columns:
        return {
            "summary": pd.DataFrame(),
            "profiles": pd.DataFrame(),
            "churn_table": pd.DataFrame(),
            "personas": {},
            "insights": [],
        }

    df = result_df.copy()
    feature_cols = _available_features(df)

    agg_map = {
        "customer_count": ("cluster", "size"),
        "predicted_churn_rate_pct": ("Churn_Label", lambda x: round(float(x.mean()) * 100, 2)),
        "avg_churn_probability_pct": ("Churn Probability", lambda x: round(float(x.mean()) * 100, 2)),
    }

    for feature in feature_cols:
        agg_map[feature] = (feature, "mean")

    summary = df.groupby("cluster").agg(**agg_map).reset_index()
    summary["segment_share_pct"] = (summary["customer_count"] / len(df) * 100).round(2)

    overall_source_cols = ["Churn_Label", "Churn Probability", *feature_cols]
    overall = df[overall_source_cols].mean(numeric_only=True)

    personas = {}
    for idx, row in summary.iterrows():
        persona_name = _persona_name(row, overall)
        summary.at[idx, "persona_name"] = persona_name
        summary.at[idx, "risk_level"] = _risk_from_churn(row["predicted_churn_rate_pct"])
        personas[int(row["cluster"])] = {
            "cluster_id": int(row["cluster"]),
            "persona_name": persona_name,
            "risk_level": summary.at[idx, "risk_level"],
            "risk_badge": _risk_badge(summary.at[idx, "risk_level"]),
            "key_characteristics": _feature_highlights(row, overall),
            "behavioral_interpretation": _behavior_interpretation(row, overall),
            "recommended_actions": _recommended_actions(row, overall),
            "customer_count": int(row["customer_count"]),
            "segment_share_pct": float(row["segment_share_pct"]),
            "predicted_churn_rate_pct": float(row["predicted_churn_rate_pct"]),
            "avg_churn_probability_pct": float(row["avg_churn_probability_pct"]),
        }

    display_cols = [
        "cluster",
        "persona_name",
        "customer_count",
        "segment_share_pct",
        "predicted_churn_rate_pct",
        "avg_churn_probability_pct",
        "risk_level",
        *feature_cols,
    ]
    summary = summary[display_cols].sort_values("predicted_churn_rate_pct", ascending=False).reset_index(drop=True)

    profiles = summary.copy()
    churn_table = summary[
        [
            "persona_name",
            "customer_count",
            "segment_share_pct",
            "predicted_churn_rate_pct",
            "avg_churn_probability_pct",
            "risk_level",
        ]
    ].copy()

    insights = _portfolio_insights(summary, overall)

    action_table = []
    for persona in personas.values():
        actions = persona["recommended_actions"]
        action_table.append(
            {
                "Persona": persona["persona_name"],
                "Risk Level": persona["risk_level"],
                "Primary Action": actions["primary_action"],
                "Targeting Rule": actions["targeting_rule"],
                "Owner": actions["owner"],
                "Retention Strategy": actions["retention_strategy"],
                "Marketing Action": actions["marketing_action"],
                "Product Improvement": actions["product_improvement"],
            }
        )

    return {
        "summary": summary,
        "profiles": profiles,
        "churn_table": churn_table,
        "personas": personas,
        "action_table": pd.DataFrame(action_table),
        "insights": insights,
    }
