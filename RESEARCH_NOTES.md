# OPPW+Carlos Optimization Research Notes

Weekly TQQQ strategy: enter Mon open, exit Fri close, with profit targets (TP_A, TP_C),
close-based stop, optional weakness mode. BIL treasury sweep when flat.
Backtest: 2010–2026, $100k, full exposure.

---

## Baseline (Original Carlos Defaults)

TP_A=8.1%, TP_C=2.5%, Stop=-1.3%, weakness=True, Mon open, Fri close.
**52.8% CAGR, 0.98 Sharpe, -47.7% MaxDD, 837 trades, 56.5% win rate.**

---

## Controlled Single-Parameter Sweeps

All sweeps freeze baseline, vary one parameter only. Delta = change vs baseline.

### TP_C — biggest mover (+9.3pp)

| TP_C | CAGR | Sharpe | MaxDD | Delta |
|------|------|--------|-------|-------|
| 0.5% | 49.1% | 0.96 | 37.7% | -3.7pp |
| **2.5%** | **52.8%** | **0.98** | **47.7%** | **baseline** |
| 4.0% | 55.9% | 1.01 | 45.3% | +3.1pp |
| **6.0%** | **62.1%** | **1.08** | **40.9%** | **+9.3pp** |

Higher TP_C lets winners run further. Also improves Sharpe and reduces drawdown.

### Weakness Mode (+4.5pp)

| Mode | CAGR | Sharpe | Delta |
|------|------|--------|-------|
| True | 52.8% | 0.98 | baseline |
| **False** | **57.3%** | **0.98** | **+4.5pp** |

Weakness heuristic exits trades that would have recovered.

### Stop Trigger (+3.0pp)

| Stop | CAGR | Sharpe | MaxDD | Delta |
|------|------|--------|-------|-------|
| **-0.3%** | **55.8%** | **1.08** | **39.4%** | **+3.0pp** |
| -0.5% | 51.8% | 1.00 | 41.6% | -1.0pp |
| -1.3% | 52.8% | 0.98 | 47.7% | baseline |
| -2.0% | 42.6% | 0.78 | 46.2% | -10.1pp |
| -5.0% | 41.9% | 0.73 | 42.6% | -10.8pp |

Tight stop (-0.3%) best. Wide stops (-2% to -5%) destroy returns.

### TP_A — already optimal

| TP_A | CAGR | Delta |
|------|------|-------|
| 5.0% | 39.6% | -13.2pp |
| **8.1%** | **52.8%** | **baseline** |
| 10.0% | 51.7% | -1.1pp |
| 15.0% | 49.5% | -3.3pp |

### Entry Day — Monday is non-negotiable

Mon=52.8%, Tue=13.4% (-39pp), Wed=-1.4% (-54pp).

### Entry Timing — open is best

open=52.8%, 9:35=51.4% (-1.4pp), 10:00=48.4% (-4.4pp). No microstructure edge.

### Exit Timing — negligible

close=52.8%, 15:30=53.0% (+0.2pp), 15:55=53.0% (+0.2pp). No MOC effect.

---

## Summary (ranked by controlled impact)

| # | Parameter | Change | CAGR Delta | Sharpe Delta |
|---|-----------|--------|------------|--------------|
| 1 | TP_C | 2.5% -> 6% | **+9.3pp** | +0.10 |
| 2 | weakness | True -> False | **+4.5pp** | 0 |
| 3 | Stop | -1.3% -> -0.3% | **+3.0pp** | +0.10 |
| 4 | TP_A | 8.1% | 0 (optimal) | — |
| 5 | Entry day | Monday | 0 (optimal) | — |
| — | Entry timing | open/9:35/vwap | -1 to -4pp | noise |
| — | Exit timing | close/15:30/15:55 | ~0pp | noise |
| — | Re-entry | cd=0/1/2 | -8 to +2pp | see below |
| 4 | Weekend hold | always | **+2.1pp** | +0.02 |

Note: Optuna optimizer breakdowns by categorical (Runs 1-3) were misleading —
they conflated parameter quality with timing quality. Always verify with
single-parameter sweeps.

### Midweek Re-entry (after TP_C / STOP)

Allow 1 re-entry per week after early exit (TP_C or STOP only, not TP_A).

| Cooldown | CAGR | Sharpe | MaxDD | Trades | Tr/yr | Delta |
|----------|------|--------|-------|--------|-------|-------|
| **none** | **52.8%** | **0.98** | **47.7%** | **837** | **52** | **baseline** |
| 0 days | 44.7% | 0.73 | 46.4% | 1123 | 70 | -8.0pp |
| 1 day | 54.7% | 0.99 | 47.4% | 888 | 55 | +1.9pp |
| 2 days | 52.8% | 0.98 | 47.7% | 837 | 52 | +0.0pp |

- cd=0: Immediate re-entry is destructive. 286 extra trades, mostly losers.
- cd=1: +1.9pp from 51 extra trades (3/yr), 58.8% win rate, 0.5% avg return. Thin edge.
- cd=2: No re-entries possible (by the time cooldown passes, it's Friday).

Verdict: Not a strong lever. The cd=1 edge is too thin to rely on.

### Weekend Hold (selective EOW exit)

Baseline always exits at Friday close (EOW). This tests: what if we remove that
forced exit and let positions run until TP or stop fires naturally?

"always hold" = remove the Friday close exit. Positions still exit via TP_A,
TP_C, or stop — just not on an arbitrary calendar boundary. If flat (after TP
or stop fires mid-week), no re-entry until the following Monday open.

| Mode | CAGR | Sharpe | MaxDD | Trades | Delta |
|------|------|--------|-------|--------|-------|
| **never** | **52.8%** | **0.98** | **47.7%** | **837** | **baseline** |
| **always** | **54.9%** | **1.00** | **38.5%** | **649** | **+2.1pp** |
| profitable | 52.7% | 0.97 | 41.9% | 675 | -0.1pp |
| sma20 | 52.1% | 0.95 | 38.5% | 675 | -0.7pp |
| sma50 | 53.2% | 0.97 | 39.8% | 672 | +0.4pp |

- "always" is the clear winner: +2.1pp CAGR, +0.02 Sharpe, **-9.2pp MaxDD**.
- Fewer trades (649 vs 837) — avoids unnecessary exit/re-enter cycles.
- More TP_A fills (211 vs 144) — longer holds give time for big target.
- 75% of trades still close within the week. Median hold = 2 days. Max = 62 days.
- Trend filters (profitable/sma20/sma50) add no value over unconditional hold.

---

## TODO

- [ ] Test stacked improvements (TP_C=6% + weakness=False + Stop=-0.3%)
- [ ] Whipsaw analysis on -0.3% stop
- [ ] Sub-period robustness check on TP_C=6%
- [ ] Walk-forward / out-of-sample validation
