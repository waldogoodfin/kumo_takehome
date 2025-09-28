from fastapi import APIRouter, Depends

from app.schemas.search import SearchQuery, SearchResponse
from app.services.search import search_catalog

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search_products(query: SearchQuery) -> SearchResponse:
    return search_catalog(query)
