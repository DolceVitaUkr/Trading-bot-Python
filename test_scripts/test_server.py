"""Test server to check UI data display"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from datetime import datetime, timedelta
import json
from pathlib import Path

app = FastAPI()

# Get the directory path for templates and static files
ui_dir = os.path.join(os.path.dirname(__file__), "tradingbot", "ui")
templates_dir = os.path.join(ui_dir, "templates")
static_dir = os.path.join(ui_dir, "static")

# Setup templates and static files
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mock data for testing
activity_log = [
    {"timestamp": datetime.now().isoformat(), "source": "SYSTEM", "message": "Trading bot started", "type": "info"},
    {"timestamp": datetime.now().isoformat(), "source": "CRYPTO", "message": "Connected to exchange", "type": "success"},
    {"timestamp": datetime.now().isoformat(), "source": "PAPER", "message": "Paper trading active", "type": "warning"}
]

# Load paper trader states if they exist
def load_paper_trader_state(asset_type):
    state_file = Path(f"tradingbot/state/paper_trader_{asset_type}.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return None

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/stats/global")
def get_global_stats():
    """Get global statistics across all assets."""
    # Load actual data from paper traders
    total_paper_wallet = 0
    active_assets = 0
    total_positions = 0
    
    for asset in ['crypto', 'futures', 'forex']:
        state = load_paper_trader_state(asset)
        if state:
            total_paper_wallet += state.get('balance', 0)
            if state.get('positions'):
                active_positions = [p for p in state['positions'] if p.get('status') == 'open']
                if active_positions:
                    active_assets += 1
                    total_positions += len(active_positions)
    
    return {
        "total_pnl": total_paper_wallet - 3000,  # Assuming 1000 start for each
        "total_paper_wallet": total_paper_wallet,
        "total_live_wallet": 0,
        "active_assets": active_assets,
        "total_positions": total_positions,
        "system_online": True
    }

@app.get("/activity/recent")
def get_recent_activity():
    """Get recent bot activity."""
    return {"activities": activity_log}

@app.get("/asset/{asset}/status")
def get_asset_status(asset: str):
    """Get status for a specific asset."""
    # Load paper trader state
    paper_state = load_paper_trader_state(asset)
    
    paper_balance = 1000.0
    paper_history = []
    
    if paper_state:
        paper_balance = paper_state.get('balance', 1000.0)
        # Create history from pnl_history if available
        if 'pnl_history' in paper_state:
            paper_history = paper_state['pnl_history'][-20:]  # Last 20 points
        
        # If not enough history, generate some mock data points
        if len(paper_history) < 5:
            import random
            base_balance = paper_state.get('starting_balance', 1000.0)
            for i in range(10):
                timestamp = datetime.now() - timedelta(hours=10-i)
                variance = random.uniform(-50, 50)
                paper_history.append({
                    "timestamp": timestamp.isoformat(),
                    "balance": base_balance + variance + (i * 2)
                })
    
    return {
        "connection_status": "connected",
        "paper_trading_active": True,
        "live_trading_active": False,
        "paper_wallet": {
            "balance": paper_balance,
            "pnl": paper_balance - 1000,
            "pnl_percent": ((paper_balance - 1000) / 1000) * 100,
            "history": paper_history
        },
        "live_wallet": {
            "balance": 0,
            "pnl": 0,
            "pnl_percent": 0,
            "history": []
        }
    }

@app.get("/asset/{asset}/positions")
def get_asset_positions(asset: str):
    """Get current positions for a specific asset."""
    # Load paper trader state
    paper_state = load_paper_trader_state(asset)
    
    paper_positions = []
    if paper_state and 'positions' in paper_state:
        paper_positions = [p for p in paper_state['positions'] if p.get('status') == 'open']
    
    # Add mock data if no real positions
    if not paper_positions and asset == 'crypto':
        paper_positions = [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "size": 0.001,
                "entry_price": 65000,
                "current_price": 66000,
                "pnl": 1.0,
                "pnl_pct": 1.54,
                "take_profit": 70000,
                "stop_loss": 63000,
                "timestamp": datetime.now().isoformat(),
                "status": "open"
            }
        ]
    
    paper_daily_pnl = sum(p.get('pnl', 0) for p in paper_positions)
    
    return {
        "paper_trading": {
            "positions": paper_positions,
            "daily_pnl": paper_daily_pnl
        },
        "live_trading": {
            "positions": [],
            "daily_pnl": 0
        }
    }

@app.get("/asset/{asset}/strategies")
def get_asset_strategies(asset: str):
    """Get strategy development status for a specific asset."""
    # Load strategies from file
    strategies_file = Path("tradingbot/state/strategies/strategies.json")
    strategies_data = {"strategies": {}}
    
    if strategies_file.exists():
        with open(strategies_file, 'r') as f:
            strategies_data = json.load(f)
    
    # Count strategies by status for this asset
    developing = 0
    pending_validation = 0
    validated = 0
    live = 0
    
    for strategy_id, strategy in strategies_data.get("strategies", {}).items():
        if strategy.get("asset_class", "").lower() == asset:
            status = strategy.get("status", "developing")
            if status == "developing":
                developing += 1
            elif status == "pending_validation":
                pending_validation += 1
            elif status == "validated":
                validated += 1
            elif status == "live_approved":
                live += 1
    
    total = developing + pending_validation + validated + live
    
    return {
        "summary": {
            "developing": developing,
            "pending_validation": pending_validation,
            "validated": validated,
            "live": live,
            "total": total
        }
    }

@app.post("/asset/{asset}/start/{mode}")
def start_asset_trading(asset: str, mode: str):
    """Mock endpoint to start trading."""
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "source": asset.upper(),
        "message": f"Started {mode} trading",
        "type": "success"
    })
    return {"asset": asset, "mode": mode, "status": "started"}

@app.post("/asset/{asset}/stop/{mode}")
def stop_asset_trading(asset: str, mode: str):
    """Mock endpoint to stop trading."""
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "source": asset.upper(),
        "message": f"Stopped {mode} trading",
        "type": "warning"
    })
    return {"asset": asset, "mode": mode, "status": "stopped"}

if __name__ == "__main__":
    print(f"Starting test server on http://localhost:8000")
    print("Check the UI at http://localhost:8000 to see if data displays correctly")
    uvicorn.run(app, host="0.0.0.0", port=8000)