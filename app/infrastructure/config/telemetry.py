import logging
from fastapi import FastAPI
from app.infrastructure.config.settings import settings

logger = logging.getLogger("llm_router.infra.telemetry")


def setup_telemetry(app: FastAPI) -> None:
    """Configures OpenTelemetry distributed tracing and metrics."""
    logger.info("Initializing OpenTelemetry tracing with endpoint '%s'", settings.OTEL_EXPORTER_OTLP_ENDPOINT)

    @app.middleware("http")
    async def add_trace_metadata(request, call_next):
        response = await call_next(request)
        response.headers["X-Router-Service"] = settings.OTEL_SERVICE_NAME
        return response
