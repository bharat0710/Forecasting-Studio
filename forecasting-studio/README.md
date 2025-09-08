# Forecasting Studio (structured)

This project contains a small FastAPI application that runs simple backtests and
walk-forward parameter searches on CSV price data. The repo is intentionally compact
and easy to extend with new strategies.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Project layout
- `app/` - python package with `main.py`, `utils.py`, `strategies.py`
- `data/` - sample CSVs and uploaded CSV (ignored by git)
- `tests/` - pytest tests
- `Dockerfile`, `.github/` - optional CI / deployment
