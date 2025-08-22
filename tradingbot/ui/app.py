"""Minimal FastAPI application for controlling the bot."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from tradingbot.core.runtime_controller import RuntimeController
from tradingbot.core.validation_manager import ValidationManager
from .routes.validation import router as validation_router
from .routes.diff import router as diff_router

runtime = RuntimeController()
validator = ValidationManager()


def create_app() -> FastAPI:
    app = FastAPI()
    
    # Include route modules
    app.include_router(validation_router)
    app.include_router(diff_router)

    @app.get("/status")
    def status():
        return runtime.get_state()

    @app.post("/live/{asset}/enable")
    def enable(asset: str):
        # Input validation
        if not asset or not asset.strip():
            raise HTTPException(status_code=400, detail="Asset symbol cannot be empty")
        if len(asset) > 20:
            raise HTTPException(status_code=400, detail="Asset symbol too long")
        if not asset.replace("/", "").replace("-", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid asset symbol format")
        
        try:
            runtime.enable_live(asset.upper().strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset.upper().strip(), "live": True}

    @app.post("/live/{asset}/disable")
    def disable(asset: str):
        # Input validation
        if not asset or not asset.strip():
            raise HTTPException(status_code=400, detail="Asset symbol cannot be empty")
        if len(asset) > 20:
            raise HTTPException(status_code=400, detail="Asset symbol too long")
        if not asset.replace("/", "").replace("-", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid asset symbol format")
        
        try:
            runtime.disable_live(asset.upper().strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset.upper().strip(), "live": False}

    @app.post("/kill/global/{onoff}")
    def kill(onoff: str):
        # Input validation for kill switch
        if not onoff or not onoff.strip():
            raise HTTPException(status_code=400, detail="Kill switch value cannot be empty")
        onoff_clean = onoff.strip().lower()
        if onoff_clean not in ["on", "off", "true", "false", "1", "0"]:
            raise HTTPException(status_code=400, detail="Kill switch must be 'on', 'off', 'true', 'false', '1', or '0'")
        
        kill_enabled = onoff_clean in ["on", "true", "1"]
        try:
            runtime.set_global_kill(kill_enabled)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to set kill switch: {exc}") from exc
        return {"kill_switch": runtime.get_state()["global"]["kill_switch"]}


    return app


app = create_app()
