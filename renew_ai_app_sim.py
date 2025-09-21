# renew_ai_app_sim.py
"""
AI-Powered Renewable Energy Optimization Platform (Simulation Only)
- /forecast_multi        -> simulated multi-source generation + demand
- /forecast_predict      -> simple forecasting (persistence/rolling-average)
- /optimize_multi        -> multi-source optimizer with battery and curtailment
- /predictive_maintenance-> simulated anomaly detection on telemetry
- /status, /             -> health + index.html
"""


import math
import random
from typing import List, Optional
import os
import numpy as np

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pyomo.environ as pyo

# Optional ML/anomaly libs
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


app = FastAPI(title="AI Renewable Energy Platform (Simulated)")
# Allow CORS for all origins (for local dashboard JS to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Simulation Helpers
# -----------------------
def simulate_multi_data(hours=24):
    """Simulated solar, wind, hydro, demand, battery SOC"""
    solar = [max(0.0, math.sin((i / hours) * math.pi) * 700 + random.gauss(0, 80)) for i in range(hours)]
    wind = [max(0.0, 200 + 100 * math.sin((i / 6) * math.pi) + random.gauss(0, 40)) for i in range(hours)]
    hydro = [max(0.0, 50 + random.gauss(0, 8)) for _ in range(hours)]
    demand = [300 + 120 * math.sin((i / 24) * 2 * math.pi) + 60 * math.sin((i / 12) * 2 * math.pi) + random.gauss(0, 20) for i in range(hours)]
    soc = [50.0 for _ in range(hours)]
    return solar, wind, hydro, demand, soc

# -----------------------
# Forecast Multi Endpoint
# -----------------------
class ForecastMultiRequest(BaseModel):
    horizon_hours: int = 24

@app.post("/forecast_multi")
def forecast_multi(req: ForecastMultiRequest):
    solar, wind, hydro, demand, soc = simulate_multi_data(req.horizon_hours)
    return {
        "source": "simulated",
        "horizon": req.horizon_hours,
        "solar": solar,
        "wind": wind,
        "hydro": hydro,
        "demand": demand,
        "battery_soc": soc
    }

# -----------------------
# Simple Forecasting Endpoint
# -----------------------
class ForecastPredictRequest(BaseModel):
    series: List[float]
    horizon: int = 12
    method: str = "persistence"

@app.post("/forecast_predict")
def forecast_predict(req: ForecastPredictRequest):
    series = list(req.series)
    h = req.horizon
    if len(series) == 0:
        return JSONResponse({"error": "series empty"}, status_code=400)
    method = req.method.lower()
    preds = []
    if method == "persistence":
        last = series[-1]
        preds = [last for _ in range(h)]
    elif method == "rolling_average":
        w = min(24, len(series))
        avg = sum(series[-w:]) / w
        preds = [avg for _ in range(h)]
    else:
        preds = [series[-1] for _ in range(h)]
    return {"method": method, "predictions": preds}

# -----------------------
# Multi-source Optimizer
# -----------------------
class OptimizeMultiRequest(BaseModel):
    solar: List[float]
    wind: List[float]
    hydro: List[float]
    demand: List[float]
    battery_capacity: float = 1000.0
    max_charge: float = 200.0
    max_discharge: float = 200.0
    init_soc: float = 500.0
    allow_grid_import: bool = False
    import_cost: float = 1.0
    cycle_penalty: float = 0.001

@app.post("/optimize_multi")
def optimize_multi(req: OptimizeMultiRequest):
    H = len(req.demand)
    if not (len(req.solar) == len(req.wind) == len(req.hydro) == H):
        return JSONResponse({"error": "input arrays must match lengths"}, status_code=400)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, H - 1)

    # Decision vars
    model.curtail_solar = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.curtail_wind = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.curtail_hydro = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T, bounds=(0.0, req.battery_capacity))
    if req.allow_grid_import:
        model.grid_import = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # SOC dynamics
    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == req.init_soc + m.charge[t] - m.discharge[t]
        return m.soc[t] == m.soc[t - 1] + m.charge[t] - m.discharge[t]
    model.soc_balance = pyo.Constraint(model.T, rule=soc_rule)

    # charge/discharge limits
    model.charge_limit = pyo.Constraint(model.T, rule=lambda m,t: m.charge[t] <= req.max_charge)
    model.discharge_limit = pyo.Constraint(model.T, rule=lambda m,t: m.discharge[t] <= req.max_discharge)

    # cannot curtail more than available
    model.curtail_s_limit = pyo.Constraint(model.T, rule=lambda m,t: m.curtail_solar[t] <= req.solar[t])
    model.curtail_w_limit = pyo.Constraint(model.T, rule=lambda m,t: m.curtail_wind[t] <= req.wind[t])
    model.curtail_h_limit = pyo.Constraint(model.T, rule=lambda m,t: m.curtail_hydro[t] <= req.hydro[t])

    # power balance
    def power_balance(m, t):
        gen_available = (req.solar[t] - m.curtail_solar[t]) + (req.wind[t] - m.curtail_wind[t]) + (req.hydro[t] - m.curtail_hydro[t])
        left = gen_available + m.discharge[t]
        if req.allow_grid_import:
            left += m.grid_import[t]
        right = req.demand[t] + m.charge[t]
        return left >= right
    model.power_balance = pyo.Constraint(model.T, rule=power_balance)

    # objective
    total_curtail = sum(model.curtail_solar[t] + model.curtail_wind[t] + model.curtail_hydro[t] for t in model.T)
    total_cycle = sum(model.charge[t] + model.discharge[t] for t in model.T)
    if req.allow_grid_import:
        total_import = sum(model.grid_import[t] for t in model.T)
        model.obj = pyo.Objective(expr=total_curtail + req.import_cost*total_import + req.cycle_penalty*total_cycle, sense=pyo.minimize)
    else:
        model.obj = pyo.Objective(expr=total_curtail + req.cycle_penalty*total_cycle, sense=pyo.minimize)


    # Try CBC, fallback to GLPK if not available
    solver = None
    for sname in ["cbc", "glpk"]:
        try:
            solver = pyo.SolverFactory(sname)
            if solver.available():
                break
        except Exception:
            continue
    if solver is None or not solver.available():
        return JSONResponse({"error": "No suitable solver (cbc or glpk) found. Please install one."}, status_code=500)
    solver.solve(model)

    return {
        "soc": [pyo.value(model.soc[t]) for t in model.T],
        "charge": [pyo.value(model.charge[t]) for t in model.T],
        "discharge": [pyo.value(model.discharge[t]) for t in model.T],
        "curtailment": {
            "solar": [pyo.value(model.curtail_solar[t]) for t in model.T],
            "wind": [pyo.value(model.curtail_wind[t]) for t in model.T],
            "hydro": [pyo.value(model.curtail_hydro[t]) for t in model.T],
            "total": sum([pyo.value(model.curtail_solar[t]) + pyo.value(model.curtail_wind[t]) + pyo.value(model.curtail_hydro[t]) for t in model.T])
        },
        "grid_import": [pyo.value(model.grid_import[t]) for t in model.T] if req.allow_grid_import else None
    }

# -----------------------
# Predictive Maintenance (Simulated)
# -----------------------
class PMRequest(BaseModel):
    telemetry: Optional[List[List[float]]] = None
    contamination: float = 0.05

@app.post("/predictive_maintenance")
def predictive_maintenance(req: PMRequest):
    telemetry = req.telemetry
    if not telemetry:
        telemetry = [[random.gauss(0.2,0.05), random.gauss(40,1), random.gauss(1500,30)] for _ in range(200)]
        for _ in range(5):
            idx = random.randint(20,180)
            telemetry[idx] = [random.uniform(1,2), random.uniform(60,90), random.uniform(1000,5000)]

    if not SKLEARN_OK:
        return {"warning":"sklearn not available","telemetry_len":len(telemetry)}

    X = np.array(telemetry)
    clf = IsolationForest(contamination=req.contamination, random_state=42)
    clf.fit(X)
    preds = clf.predict(X)
    anomalies = [{"index":i,"record":X[i].tolist()} for i,p in enumerate(preds) if p==-1]
    return {"anomalies": anomalies, "count": len(anomalies)}

# -----------------------
# Health + Frontend
# -----------------------
@app.get("/status")
def status():
    return {"status":"running"}

@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"msg":"AI Renewable Energy Platform (Simulated) API running"}

# -----------------------
# Run
# -----------------------
if __name__=="__main__":
    uvicorn.run("renew_ai_app_sim:app", host="0.0.0.0", port=8000, reload=True)
