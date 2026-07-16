from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from paper_rag.api.documents import router as documents_router
from paper_rag.api.chat import router as chat_router
from paper_rag.api.conversations import router as conversations_router
from paper_rag.api.health import router as health_router
from paper_rag.api.formulas import router as formulas_router
from paper_rag.api.jobs import router as jobs_router
from paper_rag.api.paper_profiles import router as paper_profiles_router
from paper_rag.api.search import router as search_router
from paper_rag.config import get_settings
from paper_rag.services.embeddings import get_embedding_provider
from paper_rag.db import SessionLocal
from paper_rag.services.formula_governance import mark_stale_formula_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    with SessionLocal() as session:
        mark_stale_formula_indexes(session)
    get_embedding_provider(get_settings())
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Paper RAG MVP", lifespan=lifespan)
    templates = Jinja2Templates(directory="src/paper_rag/templates")

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    app.mount("/static", StaticFiles(directory="src/paper_rag/static"), name="static")
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(search_router)
    app.include_router(chat_router)
    app.include_router(formulas_router)
    app.include_router(conversations_router)
    app.include_router(paper_profiles_router)
    return app


app = create_app()
