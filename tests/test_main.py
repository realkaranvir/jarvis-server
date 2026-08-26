import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from jarvis_server.app import create_app
from jarvis_server.config import load_config
from jarvis_server.dependencies import get_http_client


CONFIG = {
    "llm": {
        "default": "primary",
        "servers": {
            "primary": {
                "base_url": "http://llm.test/v1",
                "model": "test-model",
                "timeout_seconds": 10,
            },
            "secondary": {
                "base_url": "http://small.test/v1",
                "model": "small-model",
            },
        },
    },
    "stt": {
        "base_url": "http://stt.test/v1",
        "model": "test-whisper",
        "language": "en",
        "timeout_seconds": 10,
    },
}


class LLMEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_directory.name) / "config.json"
        self.config_path.write_text(json.dumps(CONFIG))

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_forwards_prompt_to_default_backend(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, "http://llm.test/v1/chat/completions")
            self.assertEqual(
                json.loads(request.content),
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Hi there"}}]},
            )

        app = create_app(self.config_path)
        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.dependency_overrides[get_http_client] = lambda: upstream_client

        with TestClient(app) as client:
            response = client.post("/llm", json={"prompt": "Hello"})

        asyncio.run(upstream_client.aclose())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "answer": "Hi there",
                "backend": "primary",
                "model": "test-model",
            },
        )

    def test_rejects_unknown_backend(self) -> None:
        app = create_app(self.config_path)

        with TestClient(app) as client:
            response = client.post(
                "/llm",
                json={"prompt": "Hello", "backend": "missing"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Unknown LLM backend 'missing'"})

    def test_maps_upstream_connection_failure_to_bad_gateway(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("unavailable", request=request)

        app = create_app(self.config_path)
        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.dependency_overrides[get_http_client] = lambda: upstream_client

        with TestClient(app) as client:
            response = client.post("/llm", json={"prompt": "Hello"})

        asyncio.run(upstream_client.aclose())
        self.assertEqual(response.status_code, 502)


class ConfigTests(unittest.TestCase):
    def test_requires_configured_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = CONFIG | {"llm": CONFIG["llm"] | {"default": "missing"}}
            path.write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "is not configured"):
                load_config(path)


class STTEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_directory.name) / "config.json"
        self.config_path.write_text(json.dumps(CONFIG))

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_forwards_audio_to_configured_provider(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url, "http://stt.test/v1/audio/transcriptions")
            self.assertIn(b'test-whisper', request.content)
            self.assertIn(b'en', request.content)
            self.assertIn(b'fake wave data', request.content)
            return httpx.Response(200, json={"text": "Hello from audio"})

        app = create_app(self.config_path)
        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.dependency_overrides[get_http_client] = lambda: upstream_client

        with TestClient(app) as client:
            response = client.post(
                "/stt",
                files={"audio": ("sample.wav", b"fake wave data", "audio/wav")},
            )

        asyncio.run(upstream_client.aclose())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"text": "Hello from audio", "model": "test-whisper"},
        )

    def test_rejects_empty_audio(self) -> None:
        app = create_app(self.config_path)

        with TestClient(app) as client:
            response = client.post(
                "/stt",
                files={"audio": ("sample.wav", b"", "audio/wav")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Audio file is empty"})


if __name__ == "__main__":
    unittest.main()
