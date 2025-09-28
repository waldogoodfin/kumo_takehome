from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class SearchQuery(BaseModel):
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(default=10, ge=1, le=50)


class ProductSummary(BaseModel):
    article_id: str
    product_name: str
    product_type: Optional[str]
    department: Optional[str]
    color: Optional[str]
    image_url: Optional[str]
    match_reason: Optional[str]


class SearchResponse(BaseModel):
    products: List[ProductSummary]
    strategy: Optional[str] = None
