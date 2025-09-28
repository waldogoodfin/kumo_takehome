from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    environment: str = Field(default="local")
    data_dir: Path = Field(default=Path(__file__).resolve().parents[3] / "data" / "hm_with_images")
    articles_path: Path | None = None
    customers_path: Path | None = None
    openai_api_key: str | None = Field(default=None)
    chat_model: str = Field(default="gpt-4.1-mini")
    search_sample_size: int = 100_000
    transactions_months: int = 6
    forecast_horizon_months: int = 3
    enable_embeddings: bool = Field(default=False)
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_sample_size: int = Field(default=7_500)
    embedding_batch_size: int = Field(default=100)

    class Config:
        env_file = ".env"
        env_prefix = "SHOPSIGHT_"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.articles_path = self.articles_path or self.data_dir / "articles" / "part-00000-63ea08b0-f43e-48ff-83ad-d1b7212d7840-c000.snappy.parquet"
        self.customers_path = self.customers_path or self.data_dir / "customers" / "part-00000-9b749c0f-095a-448e-b555-cbfb0bb7a01c-c000.snappy.parquet"


@lru_cache
def get_settings() -> Settings:
    return Settings()
