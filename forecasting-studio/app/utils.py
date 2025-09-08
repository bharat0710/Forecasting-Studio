"""Utility functions: CSV loaders, backtest and walk-forward logic, metrics."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import os
import io
from .strategies import create_strategy
from pydantic import BaseModel

# Path to data directory (project root / data)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

class BacktestReport(BaseModel):
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int

# --- CSV loaders ---
def load_csv_from_path(path: str) -> pd.DataFrame:
    # Try relative to cwd, then to DATA_DIR, then absolute
    if not os.path.isabs(path):
        candidate = os.path.join(os.getcwd(), path)
        if os.path.exists(candidate):
            path = candidate
        else:
            candidate2 = os.path.join(DATA_DIR, path)
            if os.path.exists(candidate2):
                path = candidate2
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns or 'close' not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'close' columns")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_csv_from_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    if 'timestamp' not in df.columns or 'close' not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'close' columns")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# --- Metrics and backtest ---
def compute_metrics(equity_curve: pd.Series, trade_returns: pd.Series) -> BacktestReport:
    # equity_curve: index=0..N, values = equity series (e.g., cumulative product)
    # trade_returns: per-trade returns (or per-change returns) used for win-rate/trades
    ret = equity_curve.pct_change().fillna(0)
    vol = ret.std()
    if vol == 0 or np.isnan(vol):
        sharpe = 0.0
    else:
        sharpe = (ret.mean() / vol) * np.sqrt(252)
    cum = equity_curve / equity_curve.iloc[0]
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1).min()
    wins = (trade_returns > 0).sum()
    trades = int(len(trade_returns))
    win_rate = float(wins) / trades if trades > 0 else 0.0
    total_return = float(cum.iloc[-1] - 1)
    return BacktestReport(total_return=total_return, sharpe=float(sharpe), max_drawdown=float(dd), win_rate=win_rate, trades=trades)

def run_backtest(df: pd.DataFrame, strategy_name: str, params: Dict[str, Any]):
    # Create strategy from registry and generate signals
    strat = create_strategy(strategy_name, params or {})
    sig = strat.generate_signals(df)
    d = df.merge(sig, on='timestamp', how='left')
    d['signal'] = d['signal'].fillna(0).astype(int)
    # position is previous day's signal (simple next-bar execution assumption)
    d['position'] = d['signal'].shift(1).fillna(0)
    d['ret'] = d['close'].pct_change().fillna(0)
    d['strategy_ret'] = d['position'] * d['ret']
    equity = (1 + d['strategy_ret']).cumprod()
    d['pos_change'] = d['position'].diff().abs().fillna(0)
    trade_mask = d['pos_change'] > 0
    trade_returns = d.loc[trade_mask, 'strategy_ret']
    report = compute_metrics(equity, trade_returns)
    return {'report': report.model_dump(), 'equity': equity.tolist(), 'timestamps': d['timestamp'].dt.isoformat().tolist()}

# --- Walk-forward grid search ---
from itertools import product

def _grid(param_space: Dict[str, Any]):
    keys = sorted(param_space.keys())
    values = [param_space[k] for k in keys]
    for vals in product(*values):
        yield dict(zip(keys, vals))

def run_walkforward(df: pd.DataFrame, strategy_name: str, param_space: Dict[str, Any], insample_days: int, outsample_days: int):
    df = df.sort_values('timestamp').reset_index(drop=True)
    results = []
    oos_returns = []
    oos_timestamps = []
    i = 0
    n = len(df)
    while True:
        is_start = i
        is_end = i + insample_days
        oos_end = is_end + outsample_days
        if oos_end > n:
            break
        is_df = df.iloc[is_start:is_end].reset_index(drop=True)
        oos_df = df.iloc[is_end:oos_end].reset_index(drop=True)
        best_sharpe = -1e9
        best_params = None
        for params in _grid(param_space):
            strat = create_strategy(strategy_name, params)
            sig = strat.generate_signals(is_df)
            d = is_df.merge(sig, on='timestamp', how='left')
            d['signal'] = d['signal'].fillna(0).astype(int)
            d['position'] = d['signal'].shift(1).fillna(0)
            d['ret'] = d['close'].pct_change().fillna(0)
            d['strategy_ret'] = d['position'] * d['ret']
            equity = (1 + d['strategy_ret']).cumprod()
            m = compute_metrics(equity, d['strategy_ret'])
            if m.sharpe > best_sharpe:
                best_sharpe = m.sharpe
                best_params = params
        # apply best params on OOS segment
        strat = create_strategy(strategy_name, best_params or {})
        sig = strat.generate_signals(oos_df)
        d2 = oos_df.merge(sig, on='timestamp', how='left')
        d2['signal'] = d2['signal'].fillna(0).astype(int)
        d2['position'] = d2['signal'].shift(1).fillna(0)
        d2['ret'] = d2['close'].pct_change().fillna(0)
        d2['strategy_ret'] = d2['position'] * d2['ret']
        oos_returns.extend(d2['strategy_ret'].tolist())
        oos_timestamps.extend(d2['timestamp'].dt.isoformat().tolist())
        results.append({'is_range': [str(is_df['timestamp'].iloc[0]), str(is_df['timestamp'].iloc[-1])], 'oos_range': [str(oos_df['timestamp'].iloc[0]), str(oos_df['timestamp'].iloc[-1])], 'best_params': best_params, 'best_is_sharpe': best_sharpe})
        i += outsample_days
    if len(oos_returns) == 0:
        return {'segments': results, 'oos': None, 'overfit_risk': None}
    oos_eq = (1 + pd.Series(oos_returns).fillna(0)).cumprod()
    oos_report = compute_metrics(oos_eq, pd.Series(oos_returns).fillna(0))
    is_sharpes = [seg['best_is_sharpe'] for seg in results]
    median_is = float(np.median(is_sharpes)) if is_sharpes else 0.0
    oos_sharpe = oos_report.sharpe
    overfit_risk = float(max(0.0, 1.0 - (oos_sharpe / (abs(median_is) + 1e-9))))
    return {'segments': results, 'oos': {'report': oos_report.model_dump(), 'timestamps': oos_timestamps}, 'overfit_risk': overfit_risk}
