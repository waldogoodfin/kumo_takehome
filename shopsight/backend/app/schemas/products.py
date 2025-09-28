from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Forecast(BaseModel):
    next_month: float
    confidence: str
    horizon: Optional[List[Dict[str, Any]]] = None


class CustomerSegment(BaseModel):
    segment: str
    percentage: float
    characteristics: str


class InsightPayload(BaseModel):
    product_id: str
    product_name: str
    sales_trend: str
    forecast: Forecast
    customer_segments: List[CustomerSegment]
    insights: str
    similar_products: List[Dict[str, Any]] = Field(default_factory=list)
    counterfactuals: Dict[str, Any] = Field(default_factory=dict)


class SalesDataset(BaseModel):
    label: str
    data: List[float]
    yAxisID: str


class SalesChart(BaseModel):
    labels: List[str]
    datasets: List[SalesDataset]


class SalesHistoryResponse(BaseModel):
    chart_data: SalesChart


class SimilarProduct(BaseModel):
    article_id: str
    product_name: str
    reasons: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    image_url: Optional[str] = None


class CounterfactualScenario(BaseModel):
    name: str
    description: str
    projected_revenue: float
    projected_units: float
    lift_percentage: float
    confidence: str
    key_drivers: List[str]


class CounterfactualResponse(BaseModel):
    baseline_revenue: float
    baseline_units: float
    scenarios: List[CounterfactualScenario]
