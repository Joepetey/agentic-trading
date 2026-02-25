"""Exception hierarchy — every error in the system has a typed home."""

from __future__ import annotations


class TradingError(Exception):
    """Base for all application errors."""


class ConfigError(TradingError):
    """Bad config, missing keys, invalid values."""


# ── Alpaca API errors ──────────────────────────────────────────────────


class AlpacaError(TradingError):
    """Base for all Alpaca API issues."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AlpacaAuthError(AlpacaError):
    """401/403 — bad keys or permissions. Never retry."""


class AlpacaRateLimitError(AlpacaError):
    """429 — rate limited. Retryable after backoff."""

    def __init__(
        self, message: str, *, status_code: int | None = 429, retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class AlpacaNetworkError(AlpacaError):
    """Connection/timeout errors. Retryable."""


class AlpacaAPIError(AlpacaError):
    """Other 4xx/5xx from Alpaca. Not retried by default."""


# ── Domain errors ──────────────────────────────────────────────────────


class RiskViolation(TradingError):
    """Risk veto triggered."""


class OrderError(TradingError):
    """Order submission or reconciliation failure."""


class StaleDataError(TradingError):
    """Latest bar is older than the staleness threshold."""
