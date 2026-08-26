import httpx
from fastapi import APIRouter, HTTPException, status

from jarvis_server.dependencies import ConfigDep, HttpClientDep
from jarvis_server.schemas import LLMRequest, LLMResponse


router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("")
async def ask_llm(
    body: LLMRequest,
    config: ConfigDep,
    client: HttpClientDep,
) -> LLMResponse:
    backend_name = body.backend or config.llm.default
    server = config.llm.servers.get(backend_name)
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown LLM backend '{backend_name}'",
        )

    try:
        response = await client.post(
            f"{server.base_url}/chat/completions",
            headers=server.headers,
            json={
                "model": server.model,
                "messages": [{"role": "user", "content": body.prompt}],
            },
            timeout=server.timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM backend '{backend_name}' request failed",
        ) from exc

    return LLMResponse(
        answer=response.json()["choices"][0]["message"]["content"],
        backend=backend_name,
        model=server.model,
    )
