from __future__ import annotations

from typing import Optional

from fastapi import Depends

from app.data.loader import DataRepository, DataBundle, get_data_repository

_repository: Optional[DataRepository] = None
_bundle: Optional[DataBundle] = None


def get_repository() -> DataRepository:
    global _repository
    if _repository is None:
        _repository = get_data_repository()
    return _repository


def get_data_bundle(repo: DataRepository = Depends(get_repository)) -> DataBundle:
    global _bundle
    if _bundle is None:
        _bundle = repo.load()
    return _bundle


def init_data_bundle() -> None:
    """Initialize data bundle during startup without FastAPI dependency injection."""
    global _bundle
    if _bundle is None:
        repo = get_repository()
        _bundle = repo.load()


def refresh_transactions() -> None:
    repo = get_repository()
    repo.refresh_transactions()
    global _bundle
    if _bundle is not None:
        _bundle.transactions = repo.load().transactions
