# telemetry/metrics_exporter.py
"""
Prometheus metrics exporter for the trading bot.

- Exposes error counters, latency histograms, and numeric gauges.
- Uses `prometheus_client` for the HTTP endpoint.
- All metrics are labeled to allow per-exchange, per-domain filtering.
"""

from prometheus_client import start_http_server, Counter, Histogram, Gauge
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────
# Metric definitions
# ──────────────────────────────────────────────────────────────
ERROR_COUNTER = Counter(
    "bot_errors_total",
    "Count of errors by code",
    ["code"]
)

LATENCY_HIST = Histogram(
    "bot_latency_ms",
    "Observed latency in milliseconds by endpoint",
    ["endpoint"],
    buckets=(1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000)
)

GAUGES: Dict[str, Gauge] = {}


# ──────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────
def start_prometheus(port: int = 9103) -> None:
    """
    Start the Prometheus metrics HTTP server.
    Args:
        port: Port to expose metrics on.
    """
    logger.info(f"Starting Prometheus metrics server on port {port}")
    start_http_server(port)


def incr_error(code: int) -> None:
    """
    Increment the error counter for a given error code.
    """
    try:
        ERROR_COUNTER.labels(code=str(code)).inc()
    except Exception:
        logger.exception("Failed to increment error counter")


def observe_latency(endpoint: str, ms: float) -> None:
    """
    Record observed latency for an endpoint.
    """
    try:
        LATENCY_HIST.labels(endpoint=endpoint).observe(ms)
    except Exception:
        logger.exception("Failed to observe latency")


def set_gauge(
        name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """
    Set a numeric gauge value.
    Args:
        name: Gauge metric name (will be prefixed 'bot_').
        value: Numeric value to set.
        labels: Optional labels dictionary.
    """
    try:
        full_name = f"bot_{name}"
        if full_name not in GAUGES:
            if labels:
                GAUGES[full_name] = Gauge(
                    full_name, f"Gauge {name}", list(labels.keys()))
            else:
                GAUGES[full_name] = Gauge(full_name, f"Gauge {name}")
        if labels:
            GAUGES[full_name].labels(**labels).set(value)
        else:
            GAUGES[full_name].set(value)
    except Exception:
        logger.exception("Failed to set gauge")
