from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    child_tokens: int
    child_overlap: int
    parent_tokens: int
    k_each: int
    k_fused: int
    k_final: int
    rrf_k: int
    grade_low: float = 0.35
    grade_high: float = 0.75
    cache_sim: float
    cache_ttl_seconds: int = 3600
    max_rewrites: int
    max_regenerations: int
    health_check_timeout: int

    database_url: str
    redis_url: str
    llm_base_url: str
    llm_model: str
    llm_api_key: SecretStr
    env: str
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_public_key: SecretStr | None = None
    max_upload_mb: int = 20
    chat_timeout_seconds: int = 90


@lru_cache
def get_settings() -> Settings:
    return Settings()
