import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.presentation.api.v1.router import router as api_router
from app.infrastructure.config.settings import settings
from app.infrastructure.config.telemetry import setup_telemetry

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("llm_router.main")

app = FastAPI(
    title="Gateway Dinâmico de Roteamento de Inferência (LLM Router)",
    description="API Gateway dinâmico para roteamento adaptativo de LLM entre vLLM e llama.cpp com OpenTelemetry e Prometheus.",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_telemetry(app)
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "openai_endpoint": "/v1/chat/completions",
        "cluster_status": "/api/v1/cluster",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
