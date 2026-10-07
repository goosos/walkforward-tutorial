# Walk-Forward Tutorial

**Part 2 of [_Build Your Own Quant Research System_](https://github.com/goosos/quant-toolkit)** — the article: [Walk-Forward Analysis: Stop Fooling Yourself with In-Sample Results](https://goosos.com/walk-forward-analysis) *(draft)*

Your backtest Sharpe is lying to you. This tutorial shows you how to measure the honest number with walk-forward analysis — and contributes `validation.py` to [goosos/quant-toolkit](https://github.com/goosos/quant-toolkit).

## What it does

Takes Part 1's MA(20,50) crossover on SPY and asks: how much of the in-sample Sharpe (1.03) survives on data it never saw?

**Answer from the actual run:** mean out-of-sample Sharpe **0.36** (35% of in-sample), 2 of 4 folds negative. The gap is the finding.

## Run it

```bash
pip install -r requirements.txt
python walkforward_demo.py
```

Needs internet for Yahoo Finance data (falls back to synthetic data offline).

## Files

| File | What |
|---|---|
| `validation.py` | The module: `walk_forward_splits()` + `run_walk_forward()` |
| `backtest.py` | Part 1's module (vendored so this repo runs standalone) |
| `walkforward_demo.py` | End-to-end demo: in-sample vs out-of-sample comparison |
| `diagram.png` / `diagram.webp` | Walk-forward window diagram (WebP is the web version) |
| `article.md` | Full tutorial text |

## The module

```python
from validation import walk_forward_splits, run_walk_forward

# 1-year train, half-year test, rolling windows
wf = run_walk_forward(price, entries, exits,
                      train_size=252, test_size=126, freq="1D")
print(wf[["test_start", "test_end", "sharpe", "n_trades"]])
print("mean OOS Sharpe:", wf["sharpe"].mean())
```

`validation.py` was merged into [goosos/quant-toolkit](https://github.com/goosos/quant-toolkit) as its second module. The toolkit version imports from `quant_toolkit.backtest` directly; this repo keeps a `try/except` fallback so it runs standalone.

## Tested

vectorbt 1.1.1 · Python 3.12 · yfinance 1.7.0 · Last verified: 2026-10-07
