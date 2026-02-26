"""Config loading — env vars for secrets, config.toml for everything else."""

from __future__ import annotations

import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class IngestConfig(BaseModel):
    timeframes: list[str] = Field(default_factory=lambda: ["1Day", "5Min"])
    lookback: dict[str, int] = Field(
        default_factory=lambda: {"1Day": 730, "5Min": 30},
        description="Calendar days to look back per timeframe",
    )
    feed: str = Field(default="iex", description="Alpaca data feed: 'iex' (free) or 'sip' (paid)")


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


class OrchestratorConfig(BaseModel):
    """Settings for the orchestrator decision cycle (Phase 3)."""

    max_staleness: dict[str, int] = Field(
        default_factory=lambda: {"1Day": 2880, "5Min": 30},
        description="Max staleness in minutes per timeframe before a symbol is excluded",
    )
    max_stale_pct: float = Field(
        default=0.50,
        description="If more than this fraction of symbols are stale, emit NO_TRADE",
    )
    primary_timeframe: str = Field(
        default="1Day",
        description="Timeframe used for eval_ts resolution",
    )


class StrategyEntry(BaseModel):
    """Per-strategy config from TOML.

    All keys become the strategy's params dict, except the reserved
    ``max_names`` / ``min_avg_volume`` / ``min_price`` fields which
    are extracted as per-strategy constraint overrides.
    """

    model_config = ConfigDict(extra="allow")

    max_names: int | None = Field(default=None, description="Per-strategy max symbols to signal on")
    min_avg_volume: int | None = Field(default=None, description="Per-strategy min avg daily volume")
    min_price: float | None = Field(default=None, description="Per-strategy min price filter")

    def strategy_params(self) -> dict[str, Any]:
        """Return only the tuneable strategy params (excludes constraint fields)."""
        reserved = {"max_names", "min_avg_volume", "min_price"}
        return {k: v for k, v in self.model_dump().items() if k not in reserved}

    def constraint_overrides(self) -> dict[str, Any]:
        """Return the per-strategy constraint overrides (only non-None)."""
        out: dict[str, Any] = {}
        if self.max_names is not None:
            out["max_names"] = self.max_names
        if self.min_avg_volume is not None:
            out["min_avg_volume"] = self.min_avg_volume
        if self.min_price is not None:
            out["min_price"] = self.min_price
        return out


class StrategyConfig(BaseModel):
    """Top-level ``[strategies]`` section from config.toml."""

    enabled: list[str] = Field(default_factory=list, description="Strategy IDs to run")
    entries: dict[str, StrategyEntry] = Field(
        default_factory=dict, description="Per-strategy config, keyed by strategy_id",
    )


class Settings(BaseModel):
    alpaca: AlpacaConfig
    symbols: SymbolUniverse = Field(default_factory=SymbolUniverse)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    strategies: StrategyConfig = Field(default_factory=StrategyConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)


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

    # ── Parse [strategies] section ───────────────────────────────────
    raw_strats = file_cfg.get("strategies", {})
    strat_enabled = raw_strats.get("enabled", [])
    strat_entries: dict[str, StrategyEntry] = {}
    for key, val in raw_strats.items():
        if key == "enabled":
            continue
        if isinstance(val, dict):
            strat_entries[key] = StrategyEntry(**val)

    return Settings(
        alpaca=AlpacaConfig(
            env=env,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
        ),
        symbols=SymbolUniverse(**file_cfg.get("symbols", {})),
        ingest=IngestConfig(**file_cfg.get("ingest", {})),
        risk=RiskLimits(**file_cfg.get("risk", {})),
        strategies=StrategyConfig(enabled=strat_enabled, entries=strat_entries),
        orchestrator=OrchestratorConfig(**file_cfg.get("orchestrator", {})),
    )
