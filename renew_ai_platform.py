# renew_ai_platform.py
"""
RenewAI Platform (Simulation-only) - single-file prototype

Endpoints:
- GET  /               -> serves index.html (dashboard)
- GET  /status         -> health
- POST /forecast_predict -> simple forecasting (persistence / rolling avg)
- POST /predictive_maintenance -> anomaly detection on telemetry (simulated)
- POST /step_sim       -> advance simulator by N timesteps (default 1) and return state
- POST /optimize_multi_nodes -> optimizer across nodes with transmission constraints
- GET  /state          -> current sim state
"""

import math, random, os, json, time
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pyomo.environ as pyo
import uvicorn

# Import our enhanced model manager for realistic ML predictions
try:
    from enhanced_ml_models import get_enhanced_model_manager
    ENHANCED_MODEL_MANAGER_AVAILABLE = True
    print("✅ Enhanced ML Model Manager loaded - using realistic ML predictions with variation")
except ImportError as e:
    ENHANCED_MODEL_MANAGER_AVAILABLE = False
    try:
        from model_manager import get_model_manager, predict_renewable_generation
        MODEL_MANAGER_AVAILABLE = True
        print("✅ Standard Model Manager loaded - using basic ML predictions")
    except ImportError as e2:
        MODEL_MANAGER_AVAILABLE = False
        print(f"⚠️  No Model Manager available: {e}")
        print("   Using simple simulation patterns")

# optional sklearn for PM - disabled for now
SKLEARN_OK = False
# try:
#     from sklearn.ensemble import IsolationForest
#     SKLEARN_OK = True
# except Exception:
#     SKLEARN_OK = False

app = FastAPI(title="RenewAI Platform - Digital Twin (Simulated)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Simple digital twin models
# ------------------------

@dataclass
class Battery:
    capacity: float          # energy units
    soc: float               # current state of charge
    max_charge: float        # kW
    max_discharge: float     # kW
    eff_charge: float = 0.98
    eff_discharge: float = 0.98

    def charge_energy(self, power_kw, hours=1.0):
        # actual energy added after efficiency
        energy = min(self.max_charge, power_kw) * hours
        added = energy * self.eff_charge
        new_soc = min(self.capacity, self.soc + added)
        charged = new_soc - self.soc
        self.soc = new_soc
        return charged

    def discharge_energy(self, power_kw, hours=1.0):
        energy = min(self.max_discharge, power_kw) * hours
        available = min(self.soc, energy / self.eff_discharge)  # ensure we don't go negative after efficiency
        out = available * self.eff_discharge
        self.soc = max(0.0, self.soc - available)
        return out

@dataclass
class Node:
    name: str
    solar: List[float] = field(default_factory=list)
    wind: List[float] = field(default_factory=list)
    hydro: List[float] = field(default_factory=list)
    demand: List[float] = field(default_factory=list)
    battery: Optional[Battery] = None
    telemetry: List[List[float]] = field(default_factory=list)  # for PM (vib,temp,rpm)
    mppt_enabled: bool = True
    demand_flex_share: float = 0.0  # fraction of demand that is flexible (for DR)

# Global simulation state: few nodes and transmission links
SIM_TIME = 0
HORIZON = 24

# Example: 3 nodes (A,B,C)
NODES: Dict[str, Node] = {}

# Transmission capacity matrix (kW), dict of (from,to)->capacity
TRANSMISSION = {}

def init_sim():
    global SIM_TIME, NODES, TRANSMISSION
    SIM_TIME = 0
    NODES = {}
    # create nodes
    for name in ["A", "B", "C"]:
        solar, wind, hydro, demand, soc = simulate_multi_data(HORIZON)
        batt = Battery(capacity=1000.0, soc=500.0, max_charge=200.0, max_discharge=200.0)
        node = Node(name=name, solar=solar, wind=wind, hydro=hydro, demand=demand,
                    battery=batt, telemetry=[], mppt_enabled=True,
                    demand_flex_share=0.15 if name=="C" else 0.05)  # C more flexible
        # fill telemetry with baseline values
        node.telemetry = [[random.gauss(0.2,0.03), random.gauss(40,1), random.gauss(1500,30)] for _ in range(300)]
        NODES[name] = node

    # set transmission capacities (symmetric)
    TRANSMISSION = {("A","B"):500.0, ("B","A"):500.0, ("B","C"):300.0, ("C","B"):300.0, ("A","C"):200.0, ("C","A"):200.0}

# Enhanced simulator using intelligent models when available
def simulate_multi_data(hours=24, start_time=None):
    """Generate renewable energy data using ML/baseline models or fallback patterns"""
    if start_time is None:
        start_time = datetime.now()
    
    if ENHANCED_MODEL_MANAGER_AVAILABLE:
        try:
            # Use enhanced ML model predictions with realistic variation
            manager = get_enhanced_model_manager()
            
            solar = []
            wind = []
            demand = []
            
            current_time = start_time
            for i in range(hours):
                # Get enhanced predictions with built-in variation
                solar_pred = manager.predict('solar', current_time)
                wind_pred = manager.predict('wind', current_time)  
                demand_pred = manager.predict('demand', current_time)
                
                # Enhanced models already include realistic variation
                solar.append(max(0.0, solar_pred))
                wind.append(max(0.0, wind_pred))
                demand.append(max(0.0, demand_pred))
                
                current_time += timedelta(hours=1)
            
            # Hydro with more variation
            hydro = [max(0.0, 50 + random.gauss(0, 15) + 20 * random.random()) for _ in range(hours)]
            soc = [50.0 for _ in range(hours)]
            
            return solar, wind, hydro, demand, soc
            
        except Exception as e:
            print(f"⚠️  Enhanced model prediction failed: {e}")
            print("   Trying standard model manager...")
    
    if MODEL_MANAGER_AVAILABLE:
        try:
            # Use basic intelligent model predictions
            manager = get_model_manager()
            
            solar = []
            wind = []
            demand = []
            
            current_time = start_time
            for i in range(hours):
                # Get predictions from models
                solar_pred = manager.predict('solar', current_time)
                wind_pred = manager.predict('wind', current_time)  
                demand_pred = manager.predict('demand', current_time)
                
                # Add some realistic noise
                solar.append(max(0.0, solar_pred + random.gauss(0, solar_pred * 0.15)))
                wind.append(max(0.0, wind_pred + random.gauss(0, wind_pred * 0.20)))
                demand.append(max(0.0, demand_pred + random.gauss(0, demand_pred * 0.08)))
                
                current_time += timedelta(hours=1)
            
            # Hydro is still simple since we don't have a specific model
            hydro = [max(0.0, 50 + random.gauss(0, 8)) for _ in range(hours)]
            soc = [50.0 for _ in range(hours)]
            
            return solar, wind, hydro, demand, soc
            
        except Exception as e:
            print(f"⚠️  Model prediction failed: {e}")
            print("   Falling back to simple patterns")
    
    # Fallback to simple patterns if models not available
    solar = [max(0.0, math.sin((i / hours) * math.pi) * 700 + random.gauss(0, 80)) for i in range(hours)]
    wind = [max(0.0, 200 + 100 * math.sin((i / 6) * math.pi) + random.gauss(0, 40)) for i in range(hours)]
    hydro = [max(0.0, 50 + random.gauss(0, 8)) for _ in range(hours)]
    demand = [300 + 120 * math.sin((i / 24) * 2 * math.pi) + 60 * math.sin((i / 12) * 2 * math.pi) + random.gauss(0, 20) for i in range(hours)]
    soc = [50.0 for _ in range(hours)]
    return solar, wind, hydro, demand, soc

# apply MPPT-like boost (simple factor when enabled)
def apply_mppt(solar_kw):
    # MPPT improvement: small stochastic boost up to 8%
    boost = 1.04 + random.random()*0.04
    return solar_kw * boost

# demand response: shift a fraction of flexible load (reduce now, increase later equally)
def apply_demand_response(node: Node, reduce_fraction=0.2, horizon_shift=3, t=0):
    """
    Reduce a fraction of flexible demand at current timestep t and add the shifted energy
    to t+horizon_shift (simple shifting, no loss).
    """
    if node.demand_flex_share <= 0:
        return 0.0
    flex = node.demand[t] * node.demand_flex_share
    reduction = flex * reduce_fraction
    node.demand[t] = max(0.0, node.demand[t] - reduction)
    future_t = min(len(node.demand)-1, t + horizon_shift)
    node.demand[future_t] += reduction
    return reduction

# ------------------------
# FastAPI models
# ------------------------

class ForecastPredictRequest(BaseModel):
    series: List[float]
    horizon: int = 12
    method: str = "persistence"

class PMRequest(BaseModel):
    telemetry: Optional[List[List[float]]] = None
    contamination: float = 0.05

class StepSimRequest(BaseModel):
    steps: int = 1
    apply_mppt: bool = True
    do_demand_response: bool = True
    dr_reduce_fraction: float = 0.25

class OptimizeMultiNodesRequest(BaseModel):
    horizon_index: int = 0   # which hour index to optimize (0..HORIZON-1)
    allow_grid_import: bool = False
    import_cost: float = 1.0
    cycle_penalty: float = 0.001

# ------------------------
# Endpoints
# ------------------------

@app.on_event("startup")
def startup_event():
    init_sim()

@app.get("/", response_class=HTMLResponse)
def root():
    # serve index.html if exists
    if os.path.exists("index.html"):
        with open("index.html","r",encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h3>RenewAI Platform (Simulated) - Running</h3><p>Place index.html next to this file to view dashboard.</p>")

@app.get("/status")
def status():
    return {"status":"running","sim_time": SIM_TIME, "nodes": list(NODES.keys())}

@app.post("/forecast_predict")
def forecast_predict(req: ForecastPredictRequest):
    series = list(req.series)
    if len(series) == 0:
        return JSONResponse({"error":"series empty"}, status_code=400)
    method = (req.method or "persistence").lower()
    h = req.horizon
    if method == "persistence":
        return {"method":"persistence", "predictions":[series[-1]]*h}
    elif method == "rolling_average":
        w = min(24, len(series))
        avg = sum(series[-w:])/w
        return {"method":"rolling_average", "predictions":[avg]*h}
    else:
        return {"method":"persistence", "predictions":[series[-1]]*h}

@app.post("/predictive_maintenance")
def predictive_maintenance(req: PMRequest):
    # if telemetry not provided, aggregate telemetry across nodes
    telemetry = req.telemetry
    if not telemetry:
        # gather recent telemetry rows from nodes
        telemetry = []
        for node in NODES.values():
            telemetry.extend(node.telemetry[-200:])
    if not SKLEARN_OK:
        return {"warning":"sklearn not installed; install scikit-learn to enable PM", "telemetry_len": len(telemetry)}
    
    # Simulate anomaly detection since sklearn is disabled
    X = np.array(telemetry)
    # Generate some random anomalies for simulation
    anomaly_indices = random.sample(range(len(X)), k=min(5, len(X)//20))
    anomalies = [{"index":i, "record": X[i].tolist()} for i in anomaly_indices]
    return {"anomalies": anomalies, "count": len(anomalies)}

@app.post("/step_sim")
def step_sim(req: StepSimRequest):
    """
    Advance simulation by N steps (timesteps). Each step we:
    - potentially apply MPPT to solar at each node
    - optionally apply demand response on nodes marked flexible
    - advance SIM_TIME
    Returns aggregated state for the horizon (current window)
    """
    global SIM_TIME
    steps = max(1, int(req.steps))
    results = []
    for s in range(steps):
        t = SIM_TIME % HORIZON
        step_state = {"t": t, "nodes": {}}
        for name, node in NODES.items():
            # read base generation/demand
            solar = node.solar[t]
            wind = node.wind[t]
            hydro = node.hydro[t]
            demand = node.demand[t]

            if req.apply_mppt and node.mppt_enabled:
                solar = apply_mppt(solar)
            dr_reduced = 0.0
            if req.do_demand_response:
                dr_reduced = apply_demand_response(node, reduce_fraction=req.dr_reduce_fraction, horizon_shift=3, t=t)

            # update telemetry (append small variations)
            node.telemetry.append([random.gauss(0.2,0.03), random.gauss(40,1), random.gauss(1500,30)])
            # limit telemetry buffer
            if len(node.telemetry) > 1000:
                node.telemetry = node.telemetry[-1000:]

            step_state["nodes"][name] = {
                "solar": solar, "wind": wind, "hydro": hydro, "demand": node.demand[t],
                "mppt_applied": node.mppt_enabled, "dr_reduced": dr_reduced,
                "battery_soc": node.battery.soc if node.battery else None
            }
        SIM_TIME += 1
        results.append(step_state)
    return {"steps_done": steps, "results": results, "sim_time": SIM_TIME}

@app.get("/state")
def get_state():
    """Return current simulation state with real-time model predictions"""
    t = SIM_TIME % HORIZON
    current_time = datetime.now()
    
    out = {
        "sim_time": SIM_TIME, 
        "horizon_index": t, 
        "current_datetime": current_time.isoformat(),
        "nodes": {}
    }
    
    for name, node in NODES.items():
        # Get real-time predictions if available
        if ENHANCED_MODEL_MANAGER_AVAILABLE:
            try:
                manager = get_enhanced_model_manager()
                realtime_solar = manager.predict('solar', current_time)
                realtime_wind = manager.predict('wind', current_time)
                realtime_demand = manager.predict('demand', current_time)
            except:
                realtime_solar = node.solar[t]
                realtime_wind = node.wind[t] 
                realtime_demand = node.demand[t]
        elif MODEL_MANAGER_AVAILABLE:
            try:
                manager = get_model_manager()
                realtime_solar = manager.predict('solar', current_time)
                realtime_wind = manager.predict('wind', current_time)
                realtime_demand = manager.predict('demand', current_time)
            except:
                realtime_solar = node.solar[t]
                realtime_wind = node.wind[t] 
                realtime_demand = node.demand[t]
        else:
            realtime_solar = node.solar[t]
            realtime_wind = node.wind[t]
            realtime_demand = node.demand[t]
        
        out["nodes"][name] = {
            "solar": node.solar, 
            "wind": node.wind, 
            "hydro": node.hydro, 
            "demand": node.demand,
            "realtime_solar": realtime_solar,
            "realtime_wind": realtime_wind,
            "realtime_demand": realtime_demand,
            "battery_soc": node.battery.soc if node.battery else None,
            "mppt_enabled": node.mppt_enabled,
            "demand_flex_share": node.demand_flex_share
        }
    
    return out

@app.get("/model_status")
def get_model_status():
    """Get status of ML and baseline models"""
    if ENHANCED_MODEL_MANAGER_AVAILABLE:
        try:
            manager = get_enhanced_model_manager()
            status = manager.get_model_status()
            status["available"] = True
            status["enhanced"] = True
            return status
        except Exception as e:
            return {"available": False, "enhanced": False, "error": str(e)}
    elif MODEL_MANAGER_AVAILABLE:
        try:
            manager = get_model_manager()
            status = manager.get_model_status()
            status["available"] = True
            status["enhanced"] = False
            return status
        except Exception as e:
            return {"available": False, "enhanced": False, "error": str(e)}
    else:
        return {
            "available": False,
            "enhanced": False,
            "message": "No Model Manager loaded - using simple simulation patterns",
            "ml_models_loaded": [],
            "baseline_models_loaded": [],
            "using_baseline": {"solar": False, "wind": False, "demand": False}
        }

@app.post("/optimize_multi_nodes")
def optimize_multi_nodes(req: OptimizeMultiNodesRequest):
    """
    Optimize dispatch at a single horizon index across nodes.
    Decision vars per node: charge, discharge, curtail_s, curtail_w, curtail_h, import/export to other nodes via transmission (net export variable)
    Transmission modeled with simple net export variables and link capacity constraints.
    """
    idx = int(req.horizon_index) % HORIZON

    # prepare arrays
    nodes = list(NODES.keys())
    K = len(nodes)
    solar = {n: NODES[n].solar[idx] for n in nodes}
    wind = {n: NODES[n].wind[idx] for n in nodes}
    hydro = {n: NODES[n].hydro[idx] for n in nodes}
    demand = {n: NODES[n].demand[idx] for n in nodes}
    battery = {n: NODES[n].battery for n in nodes}

    # Create model
    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=nodes)
    model.n = model.N

    # Vars
    model.curtail_s = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.curtail_w = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.curtail_h = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.net_export = pyo.Var(model.N)  # positive => export to network, negative => import

    # Bounds: curtail <= gen
    def curt_s_bound(m,n): return m.curtail_s[n] <= solar[n]
    def curt_w_bound(m,n): return m.curtail_w[n] <= wind[n]
    def curt_h_bound(m,n): return m.curtail_h[n] <= hydro[n]
    model.curt_s_b = pyo.Constraint(model.N, rule=curt_s_bound)
    model.curt_w_b = pyo.Constraint(model.N, rule=curt_w_bound)
    model.curt_h_b = pyo.Constraint(model.N, rule=curt_h_bound)

    # battery limits per node
    def charge_limit(m,n):
        bat = battery[n]
        return m.charge[n] <= (bat.max_charge if bat else 0.0)
    def discharge_limit(m,n):
        bat = battery[n]
        return m.discharge[n] <= (bat.max_discharge if bat else 0.0)
    model.charge_lim = pyo.Constraint(model.N, rule=charge_limit)
    model.discharge_lim = pyo.Constraint(model.N, rule=discharge_limit)

    # power balance per node: gen - curtail + discharge + net_import >= demand + charge
    def power_balance(m,n):
        gen = (solar[n]-m.curtail_s[n]) + (wind[n]-m.curtail_w[n]) + (hydro[n]-m.curtail_h[n])
        # net_import = -net_export
        net_import = -m.net_export[n]
        return gen + m.discharge[n] + net_import >= demand[n] + m.charge[n]
    model.balance = pyo.Constraint(model.N, rule=power_balance)

    # Transmission capacities: net exports must be feasible given TRANSMISSION links.
    # We'll enforce that total exports from node <= sum outgoing link capacities, and total imports likewise.
    def trans_out_limit(m,n):
        cap = 0.0
        for (i,j),c in TRANSMISSION.items():
            if i == n:
                cap += c
        return m.net_export[n] <= cap
    def trans_in_limit(m,n):
        cap = 0.0
        for (i,j),c in TRANSMISSION.items():
            if j == n:
                cap += c
        return -m.net_export[n] <= cap
    model.trans_out = pyo.Constraint(model.N, rule=trans_out_limit)
    model.trans_in = pyo.Constraint(model.N, rule=trans_in_limit)

    # global balance: sum net_export == 0 (conservation)
    model.global_balance = pyo.Constraint(expr=sum(model.net_export[n] for n in model.N) == 0)

    # Objective: minimize curtail + import cost (treated as positive net_import) + small cycle penalty
    cycle_pen = req.cycle_penalty
    def obj_expr(m):
        total_curt = sum(m.curtail_s[n] + m.curtail_w[n] + m.curtail_h[n] for n in m.N)
        total_cycle = sum(m.charge[n] + m.discharge[n] for n in m.N)
        # cost for net imports: if net_export[n] < 0, that's import; approximate cost = -min(net_export,0)
        import_cost = sum((-pyo.min(0, m.net_export[n])) for n in m.N) if req.allow_grid_import else 0
        return total_curt + req.import_cost * import_cost + cycle_pen * total_cycle
    model.obj = pyo.Objective(rule=obj_expr, sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory("cbc")
    solver.solve(model)

    # Extract results
    out = {}
    for n in nodes:
        out[n] = {
            "curtail_solar": float(pyo.value(model.curtail_s[n])),
            "curtail_wind": float(pyo.value(model.curtail_w[n])),
            "curtail_hydro": float(pyo.value(model.curtail_h[n])),
            "charge": float(pyo.value(model.charge[n])),
            "discharge": float(pyo.value(model.discharge[n])),
            "net_export": float(pyo.value(model.net_export[n])),
            "battery_soc": battery[n].soc if battery[n] else None
        }
    return {"horizon_index": idx, "results": out}

# ------------------------
# Utility: small index.html to visualize quickly (served at /)
# ------------------------
# Provide a minimal HTML file if none exists in folder
DEFAULT_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>RenewAI Sim Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script></head><body>
<h2>RenewAI Simulation Dashboard (minimal)</h2>
<button onclick="step()">Step 1</button>
<button onclick="step5()">Step 5</button>
<pre id="out"></pre>
<script>
async function step(n=1){
  const res = await fetch('/step_sim', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({steps:n})});
  const j = await res.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
function step5(){ step(5); }
</script>
</body></html>"""

# If index.html not present, write default for convenience
if not os.path.exists("index.html"):
    with open("index.html","w",encoding="utf-8") as f:
        f.write(DEFAULT_HTML)

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    # startup init
    init_sim()
    uvicorn.run("renew_ai_platform:app", host="0.0.0.0", port=8000, reload=True)
