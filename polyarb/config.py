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

    # --- Weather (METAR-driven) strategy ---
    weather_enabled: bool = False
    weather_cities: list[str] = Field(default_factory=list)
    weather_discovery_interval_s: int = 600
    weather_max_open_events: int = 8
    weather_per_event_budget_usd: float = 30.0
    weather_max_price: float = 0.40
    weather_log_filename_prefix: str = "weather_trades"
    # METAR polling cadence (adaptive: fast in HH:25-40 / HH:55-10 UTC windows).
    weather_metar_fast_period_s: int = 20
    weather_metar_slow_period_s: int = 180
    weather_ogimet_fallback: bool = False

    # --- 2-bucket forecast pre-entry ---
    weather_pair_entry_enabled: bool = True
    weather_open_meteo_host: str = "https://api.open-meteo.com"
    weather_ensemble_host: str = "https://ensemble-api.open-meteo.com"
    # Robust gate: Monte-Carlo shows σ≤1.0 / prob≥0.92 keeps realised hit-rate
    # ≥ 91% even when the ensemble is 25% under-dispersive (see backtest.py).
    weather_max_sigma_c: float = 1.0          # skip events with higher uncertainty
    weather_min_pair_prob: float = 0.92       # only enter if combined P(pair) ≥ this
    weather_pair_primary_max_price: float = 0.45
    weather_pair_secondary_max_price: float = 0.25
    weather_pair_primary_budget_share: float = 0.6
    weather_pair_lead_days_max: int = 3       # don't pre-enter > N days before resolution
    weather_live_enabled: bool = False
    polygon_private_key: str | None = None
    polygon_proxy_address: str | None = None

    @field_validator("assets", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v

    @field_validator("weather_cities", mode="before")
    @classmethod
    def _split_cities_csv(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v


def load() -> Settings:
    s = Settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    return s
