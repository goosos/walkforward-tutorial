"""
walkforward_demo.py — Goosos tutorial #2 companion code.

Takes Part 1's MA(20,50) crossover on SPY and asks the honest question:
how much of that in-sample Sharpe survives on data it never saw?

Run:  python walkforward_demo.py
Needs: vectorbt, yfinance, pandas, numpy, matplotlib
"""
import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")
from validation import walk_forward_splits, run_walk_forward
try:
    from quant_toolkit.backtest import run_backtest, ma_crossover_signals, summary
except ImportError:  # standalone tutorial repo
    from backtest import run_backtest, ma_crossover_signals, summary

SYMBOL = "SPY"
FAST, SLOW = 20, 50
TRAIN, TEST, STEP = 252, 126, 63  # 1y train / half-year test, rolling


def load_data():
    """~3 years of daily SPY closes. Falls back to synthetic data if offline."""
    try:
        import yfinance as yf
        df = yf.download(SYMBOL, period="3y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError("empty download")
        close = df["Close"].iloc[:, 0] if df["Close"].ndim > 1 else df["Close"]
        close = close.dropna()
        print(f"Data: {len(close)} daily bars of {SYMBOL} "
              f"({close.index[0].date()} -> {close.index[-1].date()})")
        return close
    except Exception as e:
        print(f"yfinance failed ({e}); using SYNTHETIC data instead.")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0004, 0.012, 756)
        idx = pd.date_range("2023-10-06", periods=756, freq="B")
        return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)


def main():
    price = load_data()
    entries, exits = ma_crossover_signals(price, FAST, SLOW)

    # --- In-sample: the Part 1 number, full period, one backtest ---
    pf_full = run_backtest(price, entries, exits, freq="1D")
    ins = summary(pf_full)
    print("\n=== IN-SAMPLE (full period, Part 1 method) ===")
    for k, v in ins.items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")

    # --- Out-of-sample: walk-forward, test segments only ---
    wf = run_walk_forward(price, entries, exits, TRAIN, TEST, STEP,
                          anchored=False, freq="1D")
    print(f"\n=== OUT-OF-SAMPLE (walk-forward, {len(wf)} folds, "
          f"train={TRAIN}d test={TEST}d) ===")
    print(wf[["test_start", "test_end", "total_return",
              "sharpe", "max_drawdown", "n_trades"]].to_string(index=False))
    print(f"\n  mean OOS Sharpe   {wf['sharpe'].mean():.4f}")
    print(f"  median OOS Sharpe {wf['sharpe'].median():.4f}")
    print(f"  OOS Sharpe < 0 in {int((wf['sharpe'] < 0).sum())}/{len(wf)} folds")

    # --- The honest comparison ---
    print("\n=== THE GAP ===")
    print(f"  In-sample Sharpe : {ins['sharpe']:.2f}")
    print(f"  Mean OOS Sharpe  : {wf['sharpe'].mean():.2f} "
          f"({100 * wf['sharpe'].mean() / ins['sharpe']:.0f}% of in-sample)")

    # --- Diagram: window layout (also saved for the article) ---
    fig, ax = plt.subplots(figsize=(10, 3))
    n = len(price)
    for i, (tr, te) in enumerate(
            walk_forward_splits(n, TRAIN, TEST, STEP, anchored=False)):
        if i >= 4:
            break
        y = 3 - i
        ax.barh(y, len(tr), left=tr[0], height=0.6, color="#4C78A8",
                label="train" if i == 0 else "")
        ax.barh(y, len(te), left=te[0], height=0.6, color="#F58518",
                label="test (out-of-sample)" if i == 0 else "")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([f"fold {i+1}" for i in range(4)])
    ax.set_xlabel("bar index")
    ax.set_title(f"Walk-forward windows (train={TRAIN}, test={TEST}, "
                 f"step={STEP}, rolling)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("diagram.png", dpi=110)
    print("\nSaved diagram.png")


if __name__ == "__main__":
    main()
