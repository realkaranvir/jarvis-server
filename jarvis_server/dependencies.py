from typing import Annotated

import httpx
from fastapi import Depends
from starlette.requests import HTTPConnection

from jarvis_server.config import AppConfig


def get_config(connection: HTTPConnection) -> AppConfig:
    return connection.app.state.config


def get_http_client(connection: HTTPConnection) -> httpx.AsyncClient:
    return connection.app.state.http_client


ConfigDep = Annotated[AppConfig, Depends(get_config)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
