from __future__ import annotations

from typing import List
from pydantic import BaseModel


class TopProduct(BaseModel):
    product_name: str
    revenue: float
    units_sold: int


class DashboardStats(BaseModel):
    total_products: int
    total_customers: int
    total_transactions: int
    total_revenue: float
    avg_order_value: float
    top_products: List[TopProduct]
