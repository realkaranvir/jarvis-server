import json
from collections.abc import AsyncIterator

import httpx

from jarvis_server.config import ServiceConfig, STTConfig


class ProviderError(RuntimeError):
    pass


async def transcribe(
    audio: bytes,
    filename: str,
    content_type: str,
    server: STTConfig,
    client: httpx.AsyncClient,
) -> str:
    fields = {"model": server.model}
    if server.language is not None:
        fields["language"] = server.language

    try:
        response = await client.post(
            f"{server.base_url.rstrip('/')}/audio/transcriptions",
            headers=server.headers,
            data=fields,
            files={"file": (filename, audio, content_type)},
            timeout=server.timeout_seconds,
        )
        response.raise_for_status()
        text = response.json()["text"]
        if not isinstance(text, str):
            raise TypeError("Transcription text is not a string")
        return text.strip()
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise ProviderError("STT provider request failed") from exc


async def stream_completion(
    prompt: str,
    server: ServiceConfig,
    client: httpx.AsyncClient,
) -> AsyncIterator[str]:
    try:
        async with client.stream(
            "POST",
            f"{server.base_url.rstrip('/')}/chat/completions",
            headers=server.headers,
            json={
                "model": server.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=server.timeout_seconds,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                choices = payload.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content
    except (httpx.HTTPError, json.JSONDecodeError, TypeError) as exc:
        raise ProviderError("LLM provider request failed") from exc
