import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from jarvis_server.app import create_app
from jarvis_server.config import VADConfig
from jarvis_server.dependencies import get_http_client
from jarvis_server.services.vad import UtteranceBuffer


CONFIG = {
    "llm": {
        "default": "primary",
        "servers": {
            "primary": {
                "base_url": "http://llm.test/v1",
                "model": "test-llm",
                "timeout_seconds": 10,
            }
        },
    },
    "stt": {
        "base_url": "http://stt.test/v1",
        "model": "test-stt",
        "language": "en",
        "timeout_seconds": 10,
    },
}


class FakeVAD:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = iter(decisions)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return next(self.decisions)


class UtteranceBufferTests(unittest.TestCase):
    def test_emits_utterance_after_silence(self) -> None:
        config = VADConfig(
            frame_duration_ms=30,
            prefix_padding_ms=60,
            silence_duration_ms=60,
            min_speech_duration_ms=30,
        )
        buffer = UtteranceBuffer(
            16000,
            config,
            FakeVAD([True, True, True, False, False]),
        )
        frame = b"\x01\x00" * 480

        events = buffer.feed(frame * 5)

        self.assertEqual(
            [event.type for event in events],
            ["speech.started", "speech.stopped"],
        )
        self.assertIsNotNone(events[-1].audio)
        self.assertEqual(events[-1].reason, "silence")


class ConversationEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_directory.name) / "config.json"
        self.config_path.write_text(json.dumps(CONFIG))

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_streams_transcript_timings_and_llm_text(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url == "http://stt.test/v1/audio/transcriptions":
                self.assertIn(b"RIFF", request.content)
                return httpx.Response(200, json={"text": "Hello Jarvis"})
            if request.url == "http://llm.test/v1/chat/completions":
                self.assertIn(b'"stream":true', request.content)
                return httpx.Response(
                    200,
                    text=(
                        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
                        'data: {"choices":[{"delta":{"content":"back"}}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404)

        app = create_app(self.config_path)
        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.dependency_overrides[get_http_client] = lambda: upstream_client

        with TestClient(app) as client:
            with client.websocket_connect("/conversation") as websocket:
                self.assertEqual(websocket.receive_json()["type"], "session.created")
                websocket.send_json({"type": "session.start", "sample_rate": 16000})
                self.assertEqual(websocket.receive_json()["type"], "session.ready")
                websocket.send_bytes(b"\x00\x00" * 16000)
                websocket.send_json({"type": "audio.commit"})

                events = []
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event["type"] == "response.completed":
                        break

        asyncio.run(upstream_client.aclose())
        transcript = next(e for e in events if e["type"] == "transcript.completed")
        completed = events[-1]
        deltas = [e["delta"] for e in events if e["type"] == "response.text.delta"]
        self.assertEqual(transcript["text"], "Hello Jarvis")
        self.assertEqual(deltas, ["Hello ", "back"])
        self.assertEqual(completed["text"], "Hello back")
        self.assertIn("stt", completed["timings_ms"])
        self.assertIn("llm_first_token", completed["timings_ms"])
