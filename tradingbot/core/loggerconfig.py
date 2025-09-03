"""
Central logging config: Rotating JSON logs (10MB x 7) + console.
"""
from __future__ import annotations
import logging, json, os, pathlib
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = pathlib.Path("tradingbot/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "app.log"

class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)

def setup_logging(level: str | int = None) -> None:
    # Make idempotent
    root = logging.getLogger()
    if getattr(root, "_tradingbot_logging_configured", False):
        return
    root._tradingbot_logging_configured = True

    # Level
    lvl = level if isinstance(level, int) else getattr(logging, str(level or os.getenv("LOG_LEVEL","INFO")).upper(), logging.INFO)
    root.setLevel(lvl)

    # File handler (JSON lines)
    fh = RotatingFileHandler(LOG_PATH, maxBytes=10*1024*1024, backupCount=7, encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(JsonLineFormatter())

    # Console handler (human-readable)
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))

    # Attach
    root.addHandler(fh)
    root.addHandler(ch)

def get_logger(name: str | None = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)