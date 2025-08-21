"""Minimal FastAPI application for controlling the bot."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from tradingbot.core.runtimecontroller import RuntimeController
from tradingbot.core.validationmanager import ValidationManager


runtime = RuntimeController()
validator = ValidationManager()


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/status")
    def status():
        return runtime.getstate()

    @app.post("/live/{asset}/enable")
    def enable(asset: str):
        try:
            runtime.enablelive(asset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset, "live": True}

    @app.post("/live/{asset}/disable")
    def disable(asset: str):
        runtime.disablelive(asset)
        return {"asset": asset, "live": False}

    @app.post("/kill/global/{onoff}")
    def kill(onoff: str):
        runtime.setglobalkill(onoff.lower() == "on")
        return {"kill_switch": runtime.getstate()["global"]["kill_switch"]}

    @app.get("/validation/{strategy_id}")
    def validation(strategy_id: str):
        report = validator.latest_report(strategy_id)
        if report is None:
            raise HTTPException(status_code=404, detail="report not found")
        return report

    return app


app = create_app()
