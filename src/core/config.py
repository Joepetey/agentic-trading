"""Config loading — env vars for secrets, config.toml for everything else."""

from __future__ import annotations

import os
import tomllib
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# ── Enums ──────────────────────────────────────────────────────────────

class Environment(str, Enum):
    PAPER = "paper"
    LIVE = "live"


# ── Models ─────────────────────────────────────────────────────────────

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


class AlpacaConfig(BaseModel):
    env: Environment = Environment.PAPER
    api_key: str
    api_secret: str
    base_url: str = ""

    @model_validator(mode="after")
    def _set_base_url(self) -> "AlpacaConfig":
        if not self.base_url:
            self.base_url = (
                PAPER_BASE_URL if self.env == Environment.PAPER else LIVE_BASE_URL
            )
        return self


class SymbolUniverse(BaseModel):
    symbols: list[str] = Field(default_factory=list)


class RiskLimits(BaseModel):
    max_position_pct: float = Field(
        default=0.05, description="Max fraction of portfolio in a single position"
    )
    max_portfolio_exposure_pct: float = Field(
        default=0.90, description="Max fraction of portfolio deployed"
    )
    max_daily_loss_pct: float = Field(
        default=0.02, description="Max daily drawdown before trading halts"
    )
    max_open_orders: int = Field(
        default=10, description="Max concurrent open orders"
    )


class Settings(BaseModel):
    alpaca: AlpacaConfig
    symbols: SymbolUniverse = Field(default_factory=SymbolUniverse)
    risk: RiskLimits = Field(default_factory=RiskLimits)


# ── Loading ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader — no extra dependencies."""
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        os.environ.setdefault(key, value)


def load_settings(config_path: Path | None = None) -> Settings:
    """Build Settings from .env (secrets) + env vars + config.toml (tuning)."""
    _load_dotenv(_PROJECT_ROOT / ".env")

    path = config_path or _DEFAULT_CONFIG_PATH
    file_cfg: dict = {}
    if path.exists():
        file_cfg = tomllib.loads(path.read_text())

    env = os.environ.get("TRADING_ENV", file_cfg.get("alpaca", {}).get("env", "paper"))
    api_key = os.environ.get(
        "ALPACA_API_KEY", file_cfg.get("alpaca", {}).get("api_key", "")
    )
    api_secret = os.environ.get(
        "ALPACA_API_SECRET", file_cfg.get("alpaca", {}).get("api_secret", "")
    )
    base_url = os.environ.get(
        "ALPACA_BASE_URL", file_cfg.get("alpaca", {}).get("base_url", "")
    )

    return Settings(
        alpaca=AlpacaConfig(
            env=env,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
        ),
        symbols=SymbolUniverse(**file_cfg.get("symbols", {})),
        risk=RiskLimits(**file_cfg.get("risk", {})),
    )
