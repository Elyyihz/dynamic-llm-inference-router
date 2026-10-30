from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Dynamic LLM Inference Router."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "dynamic-llm-inference-router"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    # Runtime Endpoints
    VLLM_NODE_URL: str = "http://mock-vllm:8001/v1"
    LLAMACPP_NODE_URL: str = "http://mock-llamacpp:8002/v1"

    # OpenTelemetry & Prometheus
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    OTEL_SERVICE_NAME: str = "dynamic-llm-inference-router"
    PROMETHEUS_METRICS_PATH: str = "/metrics"


settings = Settings()
