"""Strategy registry and example strategies.

Each strategy must implement `generate_signals(df) -> pd.DataFrame` and
return a DataFrame with at least ['timestamp', 'signal'] where 'signal' in {-1,0,1}.
"""
from typing import Dict, Any
import pandas as pd

# Simple registry to create strategies by name
_REGISTRY = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def create_strategy(name: str, params: Dict[str, Any]):
    if name not in _REGISTRY:
        raise ValueError(f'Unknown strategy: {name}')
    cls = _REGISTRY[name]
    return cls(**(params or {}))

class Strategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

@register('sma_cross')
class SmaCross(Strategy):
    """Simple moving-average crossover strategy with fast and slow windows."""
    def __init__(self, fast: int = 10, slow: int = 30):
        if not isinstance(fast, int) or not isinstance(slow, int):
            raise ValueError("fast and slow must be integers")
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d['sma_fast'] = d['close'].rolling(window=self.fast, min_periods=1).mean()
        d['sma_slow'] = d['close'].rolling(window=self.slow, min_periods=1).mean()
        d['signal'] = 0
        d.loc[d['sma_fast'] > d['sma_slow'], 'signal'] = 1
        d.loc[d['sma_fast'] < d['sma_slow'], 'signal'] = -1
        return d[['timestamp', 'signal']]
