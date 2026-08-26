from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from jarvis_server.config import get_config_path, load_config
from jarvis_server.routers import conversation, llm, stt, system


def create_app(config_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = load_config(config_path or get_config_path())
        async with httpx.AsyncClient() as client:
            app.state.http_client = client
            yield

    app = FastAPI(title="Jarvis Server", lifespan=lifespan)
    app.include_router(system.router)
    app.include_router(llm.router)
    app.include_router(stt.router)
    app.include_router(conversation.router)
    return app


app = create_app()
