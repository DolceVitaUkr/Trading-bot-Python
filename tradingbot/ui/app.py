"""Minimal FastAPI application for controlling the bot."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from tradingbot.core.runtime_controller import RuntimeController
from tradingbot.core.validation_manager import ValidationManager


runtime = RuntimeController()
validator = ValidationManager()


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/status")
    def status():
        return runtime.get_state()

    @app.post("/live/{asset}/enable")
    def enable(asset: str):
        try:
            runtime.enable_live(asset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset, "live": True}

    @app.post("/live/{asset}/disable")
    def disable(asset: str):
        runtime.disable_live(asset)
        return {"asset": asset, "live": False}

    @app.post("/kill/global/{onoff}")
    def kill(onoff: str):
        runtime.set_global_kill(onoff.lower() == "on")
        return {"kill_switch": runtime.get_state()["global"]["kill_switch"]}

    @app.get("/validation/{strategy_id}")
    def validation(strategy_id: str):
        report = validator.latest_report(strategy_id)
        if report is None:
            raise HTTPException(status_code=404, detail="report not found")
        return report

    return app


app = create_app()
