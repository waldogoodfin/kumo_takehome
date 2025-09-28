from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from rapidfuzz import process, fuzz

from app.schemas.search import ProductSummary, SearchQuery, SearchResponse
from app.utils.text import normalize_query
from app.state import get_data_bundle
from app.services.llm import get_llm_client
from app.services.embeddings import get_embedding_service


@dataclass
class SearchIntent:
    text: str
    filters: Dict[str, Any]
    strategy: str


def extract_intent(query: SearchQuery) -> SearchIntent:
    llm = get_llm_client()
    if not llm.enabled:
        return SearchIntent(text=query.query, filters={}, strategy="lexical")

    prompt = f"""
    You are helping with fashion product search. Interpret the query and return JSON with keys:
    - "normalized_query": string
    - "must_contain": optional list of keywords
    - "preferred_departments": optional list of department names
    - "notes": optional string summary

    Query: "{query.query}"
    Return JSON only.
    """

    response = llm.parse_json(prompt)
    if not response:
        return SearchIntent(text=query.query, filters={}, strategy="lexical")

    normalized = response.get("normalized_query", query.query)
    filters = {
        "must_contain": response.get("must_contain", []),
        "preferred_departments": response.get("preferred_departments", []),
        "notes": response.get("notes"),
    }
    strategy = "semantic" if get_embedding_service().enabled() else "llm+lexical"
    return SearchIntent(text=normalized, filters=filters, strategy=strategy)


def lexical_candidates(df: pd.DataFrame, intent: SearchIntent, limit: int) -> List[tuple[int, float]]:
    search_text = normalize_query(intent.text)
    choices = df["prod_name"].tolist()

    matches = process.extract(
        search_text,
        choices,
        scorer=fuzz.WRatio,
        limit=limit * 5,
    )

    candidates = []
    for match_name, score, idx in matches:
        article_id = int(df.iloc[idx]['article_id'])
        candidates.append((article_id, float(score) / 100.0))
    return candidates


def semantic_candidates(intent: SearchIntent, limit: int) -> List[tuple[int, float]]:
    embeddings = get_embedding_service()
    if not embeddings.enabled():
        return []
    return embeddings.query(intent.text, top_k=limit * 3)


def merge_scores(lexical: List[tuple[int, float]], semantic: List[tuple[int, float]], alpha: float = 0.6) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for article_id, score in lexical:
        scores[article_id] = max(scores.get(article_id, 0.0), score)
    for article_id, score in semantic:
        if article_id in scores:
            scores[article_id] = alpha * scores[article_id] + (1 - alpha) * score
        else:
            scores[article_id] = score * (1 - alpha)
    return scores


def build_summaries(df: pd.DataFrame, article_scores: Dict[int, float], intent: SearchIntent, limit: int) -> List[ProductSummary]:
    sorted_articles = sorted(article_scores.items(), key=lambda item: item[1], reverse=True)
    results = []
    for article_id, score in sorted_articles:
        row = df[df['article_id'] == article_id]
        if row.empty:
            continue
        row = row.iloc[0]
        reasons = [f"Hybrid score: {score:.2f}"]
        if intent.filters.get("preferred_departments") and row['department_name'] in intent.filters['preferred_departments']:
            reasons.append("Preferred department match")
        summary = ProductSummary(
            article_id=str(article_id),
            product_name=row['prod_name'],
            product_type=row.get('product_type_name'),
            department=row.get('department_name'),
            color=row.get('colour_group_name'),
            image_url=row.get('image_url'),
            match_reason="; ".join(reasons),
        )
        results.append(summary)
        if len(results) >= limit:
            break
    return results


def search_catalog(query: SearchQuery) -> SearchResponse:
    bundle = get_data_bundle()
    intent = extract_intent(query)

    lexical = lexical_candidates(bundle.articles, intent, query.limit)
    semantic = semantic_candidates(intent, query.limit) if get_embedding_service().enabled() else []
    article_scores = merge_scores(lexical, semantic)

    if not article_scores:
        article_scores = dict(lexical[: query.limit])

    products = build_summaries(bundle.articles, article_scores, intent, query.limit)
    strategy = intent.strategy if semantic else "lexical"
    return SearchResponse(products=products, strategy=strategy)
