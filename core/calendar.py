from __future__ import annotations

from datetime import date, timedelta

import pandas_market_calendars as mcal

_nyse = mcal.get_calendar("NYSE")

# Manual cache: (start_date, end_date) -> tuple of trading days
_cache: dict[tuple[date, date], tuple[date, ...]] = {}


def get_trading_days(start: date, end: date) -> tuple[date, ...]:
    """Return sorted tuple of NYSE trading days in [start, end].

    Results are cached — the expensive pandas_market_calendars call
    only runs once per unique (start, end) pair.
    """
    key = (start, end)
    if key in _cache:
        return _cache[key]
    schedule = _nyse.schedule(start_date=str(start), end_date=str(end))
    result = tuple(ts.date() for ts in schedule.index)
    _cache[key] = result
    return result


def first_trading_day_of_week(any_date: date) -> date:
    """Return the first NYSE trading day in the ISO week containing any_date."""
    monday = any_date - timedelta(days=any_date.weekday())
    friday = monday + timedelta(days=4)
    days = get_trading_days(monday, friday)
    return days[0]


def nth_trading_day_of_week(any_date: date, n: int = 0) -> date:
    """Return the nth (0-indexed) NYSE trading day in the ISO week.

    Clamps to last available day if n exceeds week length.
    """
    monday = any_date - timedelta(days=any_date.weekday())
    friday = monday + timedelta(days=4)
    days = get_trading_days(monday, friday)
    idx = min(n, len(days) - 1)
    return days[idx]


def last_trading_day_of_week(any_date: date) -> date:
    """Return the last NYSE trading day in the ISO week containing any_date."""
    monday = any_date - timedelta(days=any_date.weekday())
    friday = monday + timedelta(days=4)
    days = get_trading_days(monday, friday)
    return days[-1]


def warm_cache(start: date, end: date) -> int:
    """Pre-populate the get_trading_days cache for all weeks in [start, end].

    Makes a single bulk call to pandas_market_calendars, then fills
    the per-week cache entries directly. Returns the number of weeks cached.
    """
    all_timestamps = _nyse.schedule(
        start_date=str(start), end_date=str(end),
    ).index

    # Group by ISO week → populate cache with per-week (monday, friday) keys
    weeks: dict[tuple[int, int], list[date]] = {}
    for ts in all_timestamps:
        d = ts.date()
        iso = d.isocalendar()
        key = (iso[0], iso[1])
        weeks.setdefault(key, []).append(d)

    for days in weeks.values():
        monday = days[0] - timedelta(days=days[0].weekday())
        friday = monday + timedelta(days=4)
        _cache[(monday, friday)] = tuple(days)

    return len(weeks)
