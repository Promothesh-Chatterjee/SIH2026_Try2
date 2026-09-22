"""OpenTelemetry -> Azure Monitor integration for SmartScan EW API.

Exports traces, metrics, and logs to Azure Application Insights.

Set env var: APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def configure_telemetry(app: Any = None) -> bool:
    """Configure OpenTelemetry with Azure Monitor exporter.

    Returns True if configured successfully, False if connection string not set.
    Call this at application startup before any requests are processed.
    """
    conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if not conn_str or "InstrumentationKey=your-key" in conn_str or "your-instrumentation-key" in conn_str:
        logger.info(
            "APPLICATIONINSIGHTS_CONNECTION_STRING not set or placeholder — "
            "Azure Monitor telemetry export disabled. Set this env var in Azure deployment."
        )
        return False

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # Configure Azure Monitor exporter (handles traces, metrics, logs)
        configure_azure_monitor(connection_string=conn_str)

        # Auto-instrument FastAPI (adds spans for every HTTP request)
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)

        logger.info("Azure Monitor OpenTelemetry configured successfully")
        return True

    except ImportError as exc:
        logger.warning(
            "OpenTelemetry packages not installed: %s. Run: pip install azure-monitor-opentelemetry opentelemetry-instrumentation-fastapi",
            exc,
        )
        return False
    except Exception as exc:
        logger.error("Failed to configure Azure Monitor: %s", exc)
        return False


def record_inference_span(
    action: int,
    band: int,
    mode: int,
    q_score: float,
    decision_reason: str,
    hit: bool,
    latency_us: float,
) -> None:
    """Record a custom span for each scheduler inference decision.

    Visible in Azure Application Insights under 'Custom Events'.
    """
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("smartscan.scheduler")
        with tracer.start_as_current_span("scheduler.inference") as span:
            span.set_attribute("ew.action", int(action))
            span.set_attribute("ew.band", int(band))
            span.set_attribute("ew.mode", int(mode))
            span.set_attribute("ew.q_score", round(float(q_score), 4))
            span.set_attribute("ew.decision_reason", str(decision_reason))
            span.set_attribute("ew.hit", bool(hit))
            span.set_attribute("ew.latency_us", round(float(latency_us), 2))
    except Exception:
        pass  # Telemetry must never break inference


def record_metrics_event(pd: float, pfa: float, ir: float, step: int) -> None:
    """Record rolling FoM metrics as OpenTelemetry gauge measurements."""
    try:
        from opentelemetry import metrics

        meter = metrics.get_meter("smartscan.foms")
        meter.create_observable_gauge("ew.pd", callbacks=[lambda _: [(float(pd), {})]])
        meter.create_observable_gauge("ew.pfa", callbacks=[lambda _: [(float(pfa), {})]])
        meter.create_observable_gauge("ew.intercept_rate", callbacks=[lambda _: [(float(ir), {})]])
    except Exception:
        pass
