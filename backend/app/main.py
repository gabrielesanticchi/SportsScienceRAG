"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_dev import router as auth_dev_router
from app.api.documents import router as documents_router
from app.api.frontend_search import router as frontend_search_router
from app.api.frontend_upload import router as frontend_upload_router
from app.api.ingest import router as ingest_router
from app.api.search import router as search_router
from app.api.workspace import router as workspace_router
from app.qdrant.setup import ensure_hybrid_collection
from app.services.embedders import get_hybrid_embedder


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_hybrid_collection(recreate=False)
    get_hybrid_embedder()
    yield


app = FastAPI(
    title="Multi-Tenant Hybrid Ingestion Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_dev_router)
app.include_router(workspace_router)
app.include_router(frontend_upload_router)
app.include_router(frontend_search_router)
app.include_router(documents_router)
app.include_router(ingest_router)
app.include_router(search_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
