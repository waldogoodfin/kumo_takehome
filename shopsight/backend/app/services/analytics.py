from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from app.schemas.products import (
    InsightPayload,
    Forecast,
    CustomerSegment,
    SalesHistoryResponse,
    SalesChart,
    SalesDataset,
    CounterfactualResponse,
    CounterfactualScenario,
    SimilarProduct,
)
from app.utils.text import fallback_insight
from app.services.llm import get_llm_client
from app.state import get_data_bundle


@dataclass
class ProductContext:
    article_id: int
    product_row: pd.Series
    transactions: pd.DataFrame


def get_product_context(article_id: int) -> ProductContext:
    bundle = get_data_bundle()
    product_mask = bundle.articles['article_id'] == article_id
    product_rows = bundle.articles[product_mask]
    if product_rows.empty:
        raise ValueError("Product not found")
    product_row = product_rows.iloc[0]

    product_transactions = bundle.transactions[bundle.transactions['article_id'] == article_id]
    return ProductContext(article_id=article_id, product_row=product_row, transactions=product_transactions)


def compute_sales_trend(transactions: pd.DataFrame) -> tuple[str, pd.Series]:
    if transactions.empty:
        return "No sales data available", pd.Series(dtype=float)

    monthly_sales = transactions.copy()
    monthly_sales['month'] = monthly_sales['transaction_date'].dt.to_period('M')
    monthly_agg = monthly_sales.groupby('month')['total_amount'].sum()

    if len(monthly_agg) >= 2:
        recent_trend = monthly_agg.iloc[-1] - monthly_agg.iloc[-2]
        trend_text = "increased" if recent_trend >= 0 else "decreased"
        return f"Sales {trend_text} by ${abs(recent_trend):.2f} last month", monthly_agg
    return "Insufficient data for trend analysis", monthly_agg


def generate_forecast(monthly_sales: pd.Series) -> Forecast:
    if monthly_sales.empty:
        return Forecast(next_month=0.0, confidence="Low", horizon=[])

    avg_sales = monthly_sales.mean()
    rng = np.random.default_rng(123)
    horizon = []
    for i in range(1, 4):
        value = avg_sales * rng.uniform(0.9, 1.25)
        horizon.append({"month_offset": i, "projected_revenue": round(float(value), 2)})

    return Forecast(
        next_month=round(float(horizon[0]['projected_revenue']), 2),
        confidence="Medium" if len(monthly_sales) >= 3 else "Low",
        horizon=horizon,
    )


def mock_segments(product_row: pd.Series) -> List[CustomerSegment]:
    segments = [
        CustomerSegment(segment="Fashion Enthusiasts", percentage=42, characteristics="High engagement, frequently purchase new arrivals"),
        CustomerSegment(segment="Value Seekers", percentage=33, characteristics="Watch for promotions, respond to bundle offers"),
        CustomerSegment(segment="Trend Followers", percentage=25, characteristics="Influenced by social channels and fast trends"),
    ]
    return segments


def llm_insights(product_row: pd.Series, trend: str, forecast: Forecast) -> str:
    llm = get_llm_client()
    if not llm.enabled:
        return fallback_insight(product_row['product_type_name'], product_row['department_name'], trend)

    # Extract trend direction and magnitude
    trend_direction = "growing" if "increased" in trend.lower() else "declining" if "decreased" in trend.lower() else "stable"
    
    prompt = f"""
    You are a senior retail strategist analyzing product performance. Generate aggressive, data-driven insights.

    PRODUCT CONTEXT:
    - Product: {product_row['prod_name']}
    - Type: {product_row['product_type_name']} in {product_row['department_name']}
    - Color: {product_row['colour_group_name']}
    
    PERFORMANCE DATA:
    - Sales Trend: {trend} (momentum is {trend_direction})
    - Next Month Forecast: ${forecast.next_month:.2f} (confidence: {forecast.confidence})
    - 3-Month Outlook: {', '.join([f'${h["projected_revenue"]:.0f}' for h in forecast.horizon[:3]])}

    Generate 3 AGGRESSIVE, specific action items:
    1. Marketing/positioning move with specific tactics
    2. Inventory/pricing decision with rationale  
    3. Customer acquisition strategy with channel focus

    Be direct, use numbers, reference the trend momentum. Sound like a consultant who studied the data.
    Format as bullet points starting with action verbs.
    
    IMPORTANT: Return PLAIN TEXT only. Do NOT use markdown formatting, asterisks, or bold text. Use simple bullet points with hyphens.
    """

    result = llm.chat(prompt, temperature=0.4)
    if not result:
        return fallback_insight(product_row['product_type_name'], product_row['department_name'], trend)
    return result


def build_insights(article_id: int) -> InsightPayload:
    context = get_product_context(article_id)
    product_row = context.product_row
    trend, monthly_sales = compute_sales_trend(context.transactions)
    forecast = generate_forecast(monthly_sales)
    segments = mock_segments(product_row)

    insights_text = llm_insights(product_row, trend, forecast)

    similar = suggest_similar_products(product_row)
    counterfactuals = generate_counterfactuals(context, forecast)

    return InsightPayload(
        product_id=str(article_id),
        product_name=product_row['prod_name'],
        sales_trend=trend,
        forecast=forecast,
        customer_segments=segments,
        insights=insights_text,
        similar_products=[s.dict() for s in similar],
        counterfactuals={
            "baseline_revenue": counterfactuals.baseline_revenue,
            "baseline_units": counterfactuals.baseline_units,
            "scenarios": [c.dict() for c in counterfactuals.scenarios]
        },
    )


def get_sales_history(article_id: int) -> SalesHistoryResponse:
    context = get_product_context(article_id)
    transactions = context.transactions
    if transactions.empty:
        return SalesHistoryResponse(
            chart_data=SalesChart(labels=[], datasets=[SalesDataset(label="Revenue", data=[], yAxisID="y")])
        )

    transactions = transactions.copy()
    transactions['month'] = transactions['transaction_date'].dt.to_period('M')
    monthly = transactions.groupby('month').agg({'total_amount': 'sum', 'quantity': 'sum'}).reset_index()

    labels = [str(period) for period in monthly['month']]
    revenue_data = monthly['total_amount'].astype(float).round(2).tolist()
    units_data = monthly['quantity'].astype(int).tolist()

    return SalesHistoryResponse(
        chart_data=SalesChart(
            labels=labels,
            datasets=[
                SalesDataset(label="Revenue ($)", data=revenue_data, yAxisID="y"),
                SalesDataset(label="Units Sold", data=units_data, yAxisID="y1"),
            ],
        )
    )


def suggest_similar_products(product_row: pd.Series, limit: int = 4) -> List[SimilarProduct]:
    bundle = get_data_bundle()
    same_department = bundle.articles[bundle.articles['department_name'] == product_row['department_name']]
    candidates = same_department[same_department['article_id'] != product_row['article_id']]

    if candidates.empty:
        return []

    sampled = candidates.sample(n=min(limit, len(candidates)), random_state=product_row['article_id'])
    results: List[SimilarProduct] = []
    for _, row in sampled.iterrows():
        reasons = ["Same department"]
        if row['product_type_name'] == product_row['product_type_name']:
            reasons.append("Matching product type")
        if row['colour_group_name'] == product_row['colour_group_name']:
            reasons.append("Similar color palette")
        results.append(
            SimilarProduct(
                article_id=str(row['article_id']),
                product_name=row['prod_name'],
                reasons=reasons,
                image_url=row.get('image_url'),
            )
        )
    return results


def generate_counterfactuals(context: ProductContext, forecast: Forecast) -> CounterfactualResponse:
    baseline_revenue = max(forecast.next_month, 50.0)  # Minimum baseline for demo
    baseline_units = float(context.transactions['quantity'].mean() * 30) if not context.transactions.empty else 10.0

    # If no sales data, use product-appropriate baseline estimates
    if context.transactions.empty:
        product_type = context.product_row.get('product_type_name', '').lower()
        if 'dress' in product_type:
            baseline_revenue = 180.0
            baseline_units = 12.0
        elif 'coat' in product_type or 'jacket' in product_type:
            baseline_revenue = 320.0
            baseline_units = 8.0
        else:
            baseline_revenue = 120.0
            baseline_units = 15.0

    scenarios = [
        CounterfactualScenario(
            name="Influencer Boost",
            description="Scale creator partnerships targeting Trend Followers",
            projected_revenue=round(baseline_revenue * 1.18, 2),
            projected_units=round(baseline_units * 1.12, 1),
            lift_percentage=18.0,
            confidence="Medium",
            key_drivers=["Social amplification", "Trend follower engagement"],
        ),
        CounterfactualScenario(
            name="Bundle Offer",
            description="Bundle with complementary items at 15% discount",
            projected_revenue=round(baseline_revenue * 1.12, 2),
            projected_units=round(baseline_units * 1.15, 1),
            lift_percentage=12.0,
            confidence="Medium",
            key_drivers=["Average order value uplift", "Basket size increase"],
        ),
        CounterfactualScenario(
            name="Geo-Targeted Ads",
            description="Focus campaigns on high-conversion regions",
            projected_revenue=round(baseline_revenue * 1.08, 2),
            projected_units=round(baseline_units * 1.05, 1),
            lift_percentage=8.0,
            confidence="Low",
            key_drivers=["Localized messaging", "Regional targeting"],
        ),
    ]

    return CounterfactualResponse(
        baseline_revenue=round(baseline_revenue, 2),
        baseline_units=round(baseline_units, 1),
        scenarios=scenarios,
    )
