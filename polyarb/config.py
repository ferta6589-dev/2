from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    assets: list[str] = Field(default_factory=lambda: ["BTC"])
    profit_bps_min: int = 50
    min_time_to_settle_s: int = 10
    fee_rate_bps: int = 200

    gamma_host: str = "https://gamma-api.polymarket.com"
    clob_http_host: str = "https://clob.polymarket.com"
    clob_ws_host: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    log_dir: Path = Path("./logs")
    discovery_interval_s: int = 15

    @field_validator("assets", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v


def load() -> Settings:
    s = Settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    return s
