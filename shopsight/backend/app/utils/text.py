from __future__ import annotations

import re
from typing import Iterable


def normalize_query(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_bullet_points(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def fallback_insight(product_type: str, department: str, trend: str) -> str:
    return (
        f"This {product_type.lower()} in the {department.lower()} department currently {trend.lower()}. "
        "Consider targeted campaigns and inventory alignment to capture upcoming demand."
    )
