from fastapi import APIRouter, HTTPException

from app.schemas.products import InsightPayload, SalesHistoryResponse, CounterfactualResponse
from app.services.analytics import build_insights, get_sales_history, generate_counterfactuals, get_product_context

router = APIRouter(prefix="/product", tags=["products"])


@router.get("/{article_id}/insights", response_model=InsightPayload)
def product_insights(article_id: str) -> InsightPayload:
    try:
        return build_insights(int(article_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{article_id}/sales-history", response_model=SalesHistoryResponse)
def product_sales_history(article_id: str) -> SalesHistoryResponse:
    try:
        return get_sales_history(int(article_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{article_id}/counterfactuals", response_model=CounterfactualResponse)
def product_counterfactuals(article_id: str) -> CounterfactualResponse:
    try:
        context = get_product_context(int(article_id))
        forecast = build_insights(int(article_id)).forecast
        return generate_counterfactuals(context, forecast)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
