"""FastAPI app entrypoint for Forecasting Studio.

This file wires routes and serves a tiny single-file frontend as well.
It imports utility functions and strategies from the app package.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import os

# Local imports from package
from .utils import load_csv_from_path, load_csv_from_bytes, run_backtest, run_walkforward
from .strategies import register, SmaCross  # ensures registration of built-in strategies

app = FastAPI(title='Forecasting Studio - Structured')

# Data dir (relative to project) - the utils functions will also reference this
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
DATA_DIR = os.path.abspath(DATA_DIR)

# Small schema definitions for endpoints
class BacktestRequest(BaseModel):
    csv_path: str
    strategy_name: str
    params: Dict[str, Any] = {}

class WalkForwardRequest(BaseModel):
    csv_path: str
    strategy_name: str
    param_space: Dict[str, List[Any]]
    insample_days: int = Field(default=252, ge=1)
    outsample_days: int = Field(default=63, ge=1)

# Serve the small frontend (copied from the single-file example)
@app.get('/', response_class=HTMLResponse)
async def homepage():
    html = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Forecasting Studio (structured)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  </head>
  <body>
    <h1>Forecasting Studio (structured)</h1>
    <div>
      <label>CSV Path (relative):</label>
      <input id="csvPath" value="data/sample_prices.csv" style="width:320px" />
      <button onclick="runBacktest()">Backtest</button>
      <button onclick="runWalkforward()">Walk-Forward</button>
    </div>
    <div style="margin-top:8px">
      <label>Or Upload CSV:</label>
      <input type="file" id="csvfile" />
      <button onclick="upload()">Upload</button>
    </div>
    <p id="log"></p>
    <pre id="report"></pre>
    <canvas id="chart" width="800" height="300"></canvas>

<script>
async function upload(){
  const f = document.getElementById('csvfile').files[0];
  if(!f){ alert('choose file'); return }
  const form = new FormData(); form.append('file', f);
  document.getElementById('log').innerText = 'Uploading...';
  const res = await fetch('/upload', { method: 'POST', body: form });
  const j = await res.json();
  document.getElementById('log').innerText = 'Uploaded. Preview: ' + JSON.stringify(j.head);
}

async function runBacktest(){
  const csvPath = document.getElementById('csvPath').value;
  document.getElementById('log').innerText = 'Running backtest...';
  const res = await fetch('/backtest', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ csv_path: csvPath, strategy_name: 'sma_cross', params: { fast: 10, slow: 30 } }) });
  const j = await res.json();
  document.getElementById('report').innerText = JSON.stringify(j.report, null, 2);
  document.getElementById('log').innerText = 'Done';
  drawChart(j.timestamps, j.equity);
}

async function runWalkforward(){
  const csvPath = document.getElementById('csvPath').value;
  document.getElementById('log').innerText = 'Running walkforward...';
  const res = await fetch('/walkforward', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ csv_path: csvPath, strategy_name: 'sma_cross', param_space: { fast: [5,10,20], slow: [30,50] }, insample_days: 3, outsample_days: 1 }) });
  const j = await res.json();
  document.getElementById('report').innerText = JSON.stringify(j, null, 2);
  document.getElementById('log').innerText = 'Done';
}

let chart = null;
function drawChart(labels, data){
  const ctx = document.getElementById('chart').getContext('2d');
  if(chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{ label: 'Equity', data: data, fill: false, tension: 0.1 }]
    },
    options: { scales: { x: { display: false } } }
  });
}
</script>
  </body>
</html>"""
    return HTMLResponse(content=html)

# Upload endpoint
@app.post('/upload')
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Please upload a CSV file')
    content = await file.read()
    try:
        df = load_csv_from_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    uploaded_path = os.path.join(DATA_DIR, 'uploaded.csv')
    os.makedirs(os.path.dirname(uploaded_path), exist_ok=True)
    with open(uploaded_path, 'wb') as f:
        f.write(content)
    return {'columns': df.columns.tolist(), 'rows': min(5, len(df)), 'head': df.head(min(5, len(df))).to_dict(orient='records'), 'saved_to': uploaded_path}

# Backtest endpoint
@app.post('/backtest')
async def api_backtest(req: BacktestRequest):
    try:
        df = load_csv_from_path(req.csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = run_backtest(df, req.strategy_name, req.params or {})
    return JSONResponse(result)

# Walk-forward endpoint
@app.post('/walkforward')
async def api_walkforward(req: WalkForwardRequest):
    try:
        df = load_csv_from_path(req.csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = run_walkforward(df, req.strategy_name, req.param_space, req.insample_days, req.outsample_days)
    return JSONResponse(result)

# Optionally run with `python -m uvicorn app.main:app --reload`
if __name__ == '__main__':
    import uvicorn
    print('Starting Forecasting Studio (structured). Open http://localhost:8000')
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
