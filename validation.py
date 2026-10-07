"""
validation.py — Part 2 of "Build Your Own Quant Research System".

Walk-forward validation: the honest way to check whether a backtest
result survives contact with data it has never seen.

In-sample vs out-of-sample, in one function call.

Tutorial: https://goosos.com/walk-forward-analysis
Code: https://github.com/goosos/walkforward-tutorial
"""
import pandas as pd

try:
    # When merged into the toolkit (the real home of this module)
    from quant_toolkit.backtest import run_backtest, summary
except ImportError:  # standalone: running from the tutorial repo
    from backtest import run_backtest, summary


def walk_forward_splits(n, train_size, test_size, step=None, anchored=False):
    """
    Yield (train_idx, test_idx) integer indexers for walk-forward validation.

    Parameters
    ----------
    n : int
        Total number of bars.
    train_size : int
        Bars in each training window.
    test_size : int
        Bars in each out-of-sample test window.
    step : int, optional
        How far the window advances each fold. Defaults to test_size
        (non-overlapping test segments).
    anchored : bool
        False (default): rolling windows — the train window slides forward,
        always train_size bars. True: anchored — the train window starts at
        bar 0 every fold and grows, so later folds train on more history.

    Yields
    ------
    (train_idx, test_idx) : tuple of pd.Index
        Integer positions into the price series.
    """
    if step is None:
        step = test_size
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size and step must be positive")
    if train_size + test_size > n:
        raise ValueError(
            f"train_size ({train_size}) + test_size ({test_size}) "
            f"exceeds data length ({n})"
        )

    idx = pd.RangeIndex(n)
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n:
            break
        train_idx = idx[:train_end] if anchored else idx[start:train_end]
        test_idx = idx[train_end:test_end]
        yield train_idx, test_idx
        start += step


def run_walk_forward(price, entries, exits, train_size, test_size,
                     step=None, anchored=False, freq="1D", **bt_kwargs):
    """
    Run walk-forward validation on precomputed signals.

    Signals must already be shifted one bar (no lookahead) — see
    quant_toolkit.backtest.ma_crossover_signals. Each fold slices the
    signal series to its test window and backtests ONLY that segment,
    so every reported metric is out-of-sample.

    NOTE: parameters are held fixed across folds here. A stricter
    walk-forward re-selects parameters on each train window (that is
    selection-bias territory — covered in Part 4 on overfitting).
    Holding params fixed isolates the pure in-sample vs out-of-sample
    gap: how much of Part 1's Sharpe survives unseen data.

    Parameters
    ----------
    price : pd.Series
        Full close-price series, datetime-indexed.
    entries, exits : pd.Series of bool
        Precomputed, shifted signals aligned with price.
    train_size, test_size, step, anchored
        See walk_forward_splits.
    freq : str
        Bar frequency for annualization ("1D", "1h").
    **bt_kwargs
        Passed to run_backtest (fees, slippage, init_cash...).

    Returns
    -------
    pd.DataFrame
        One row per fold: test_start, test_end, and the summary()
        metrics computed on the out-of-sample segment only.
    """
    rows = []
    for train_idx, test_idx in walk_forward_splits(
            len(price), train_size, test_size, step, anchored):
        seg_price = price.iloc[test_idx]
        seg_entries = entries.iloc[test_idx]
        seg_exits = exits.iloc[test_idx]
        # Skip folds where the strategy never trades — Sharpe of
        # an empty segment is undefined, not zero.
        if not seg_entries.any():
            continue
        pf = run_backtest(seg_price, seg_entries, seg_exits,
                          freq=freq, **bt_kwargs)
        m = summary(pf)
        m["test_start"] = seg_price.index[0]
        m["test_end"] = seg_price.index[-1]
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["test_start", "test_end", "total_return", "sharpe",
            "max_drawdown", "win_rate", "n_trades", "profit_factor"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    cols = ["test_start", "test_end"] + [c for c in df.columns
                                         if c not in ("test_start", "test_end")]
    return df[cols]
