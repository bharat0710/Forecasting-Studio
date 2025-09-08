import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_homepage():
    r = client.get('/')
    assert r.status_code == 200
    assert 'Forecasting Studio' in r.text

def test_backtest_sample():
    payload = {"csv_path": "data/sample_prices.csv", "strategy_name": "sma_cross", "params": {"fast": 2, "slow": 3}}
    r = client.post('/backtest', json=payload)
    assert r.status_code == 200
    j = r.json()
    assert 'report' in j
    assert 'equity' in j
