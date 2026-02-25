"""Wrapped Alpaca clients — retries, error classification, structured logging."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from requests.exceptions import ConnectionError, ReadTimeout, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.core.config import AlpacaConfig, Environment
from src.core.errors import (
    AlpacaAPIError,
    AlpacaAuthError,
    AlpacaNetworkError,
    AlpacaRateLimitError,
)

log = structlog.get_logger()

# ── Error classification ───────────────────────────────────────────────


def _classify_alpaca_error(exc: APIError) -> AlpacaAuthError | AlpacaRateLimitError | AlpacaAPIError:
    """Turn an alpaca-py APIError into our typed hierarchy."""
    status = exc.status_code
    msg = str(exc)

    if status in (401, 403):
        return AlpacaAuthError(msg, status_code=status)
    if status == 429:
        retry_after = None
        if exc.response is not None:
            retry_after_hdr = exc.response.headers.get("Retry-After")
            if retry_after_hdr is not None:
                try:
                    retry_after = float(retry_after_hdr)
                except ValueError:
                    pass
        return AlpacaRateLimitError(msg, status_code=status, retry_after=retry_after)
    return AlpacaAPIError(msg, status_code=status)


def _classify_and_raise(exc: Exception) -> None:
    """Classify any exception from an Alpaca call and raise our typed version."""
    if isinstance(exc, (ConnectionError, Timeout, ReadTimeout, OSError)):
        raise AlpacaNetworkError(str(exc)) from exc
    if isinstance(exc, APIError):
        raise _classify_alpaca_error(exc) from exc
    raise AlpacaAPIError(str(exc)) from exc


# ── Retry decorator for read operations ────────────────────────────────

_RETRYABLE = (AlpacaNetworkError, AlpacaRateLimitError)

_read_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


# ── Data client ────────────────────────────────────────────────────────


class AlpacaDataClient:
    """Wrapped StockHistoricalDataClient with retries and structured logging."""

    def __init__(self, cfg: AlpacaConfig) -> None:
        self._client = StockHistoricalDataClient(cfg.api_key, cfg.api_secret)
        self._log = log.bind(client="data")

    @_read_retry
    def get_stock_bars(self, request: StockBarsRequest) -> Any:
        self._log.info(
            "fetching_bars",
            symbols=request.symbol_or_symbols,
            timeframe=str(request.timeframe),
        )
        try:
            result = self._client.get_stock_bars(request)
        except Exception as exc:
            self._log.error("bars_request_failed", error=str(exc))
            _classify_and_raise(exc)
        self._log.info("bars_fetched", row_count=len(result.df))
        return result


# ── Trading client ─────────────────────────────────────────────────────


class AlpacaTradingClient:
    """Wrapped TradingClient with retries on reads, no retry on writes."""

    def __init__(self, cfg: AlpacaConfig) -> None:
        self._client = TradingClient(
            cfg.api_key,
            cfg.api_secret,
            paper=(cfg.env == Environment.PAPER),
        )
        self._log = log.bind(client="trading")

    @_read_retry
    def get_account(self) -> Any:
        self._log.info("fetching_account")
        try:
            result = self._client.get_account()
        except Exception as exc:
            self._log.error("account_request_failed", error=str(exc))
            _classify_and_raise(exc)
        return result

    @_read_retry
    def get_open_positions(self) -> Any:
        self._log.info("fetching_positions")
        try:
            result = self._client.get_all_positions()
        except Exception as exc:
            self._log.error("positions_request_failed", error=str(exc))
            _classify_and_raise(exc)
        return result

    @_read_retry
    def get_open_orders(self) -> Any:
        self._log.info("fetching_orders")
        try:
            result = self._client.get_orders()
        except Exception as exc:
            self._log.error("orders_request_failed", error=str(exc))
            _classify_and_raise(exc)
        return result

    def submit_order(self, request: MarketOrderRequest) -> Any:
        """Submit an order. NOT retried — no accidental double-submits."""
        self._log.info(
            "submitting_order",
            symbol=request.symbol,
            side=str(request.side),
            qty=str(request.qty),
        )
        try:
            result = self._client.submit_order(request)
        except Exception as exc:
            self._log.error(
                "order_submission_failed",
                symbol=request.symbol,
                error=str(exc),
            )
            _classify_and_raise(exc)
        self._log.info("order_submitted", order_id=str(result.id), status=str(result.status))
        return result

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order by ID. NOT retried."""
        self._log.info("cancelling_order", order_id=order_id)
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as exc:
            self._log.error("order_cancel_failed", order_id=order_id, error=str(exc))
            _classify_and_raise(exc)
        self._log.info("order_cancelled", order_id=order_id)
