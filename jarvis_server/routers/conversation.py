import io
import json
import wave
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from jarvis_server.dependencies import ConfigDep, HttpClientDep
from jarvis_server.services.providers import ProviderError, stream_completion, transcribe
from jarvis_server.services.vad import SUPPORTED_SAMPLE_RATES, UtteranceBuffer


router = APIRouter(prefix="/conversation", tags=["conversation"])


class SessionStart(BaseModel):
    type: Literal["session.start"]
    sample_rate: int = 16000
    backend: str | None = None


def pcm_to_wav(audio: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)
    return output.getvalue()


async def send_error(websocket: WebSocket, stage: str, message: str) -> None:
    await websocket.send_json({"type": "error", "stage": stage, "message": message})


@router.websocket("")
async def conversation(
    websocket: WebSocket,
    config: ConfigDep,
    client: HttpClientDep,
) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "session.created",
            "audio_format": "pcm_s16le",
            "channels": 1,
            "supported_sample_rates": sorted(SUPPORTED_SAMPLE_RATES),
        }
    )

    try:
        start = SessionStart.model_validate_json(await websocket.receive_text())
        if start.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {start.sample_rate}")

        backend_name = start.backend or config.llm.default
        llm_server = config.llm.servers.get(backend_name)
        if llm_server is None:
            raise ValueError(f"Unknown LLM backend: {backend_name}")

        audio = UtteranceBuffer(start.sample_rate, config.vad)
        await websocket.send_json(
            {
                "type": "session.ready",
                "sample_rate": start.sample_rate,
                "backend": backend_name,
            }
        )

        utterance: bytes | None = None
        stop_reason = "client_commit"
        while utterance is None:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                for event in audio.feed(message["bytes"]):
                    if event.type == "speech.started":
                        await websocket.send_json({"type": "speech.started"})
                    elif event.type == "speech.stopped":
                        utterance = event.audio
                        stop_reason = event.reason or "silence"
                        break
            elif message.get("text") is not None:
                event = json.loads(message["text"])
                if event.get("type") == "audio.commit":
                    utterance = audio.commit()
                elif event.get("type") == "session.cancel":
                    await websocket.send_json({"type": "session.cancelled"})
                    return

        if not utterance:
            raise ValueError("No audio was received")

        audio_duration_ms = len(utterance) / (start.sample_rate * 2) * 1000
        processing_started = perf_counter()
        await websocket.send_json(
            {
                "type": "speech.stopped",
                "reason": stop_reason,
                "audio_duration_ms": round(audio_duration_ms, 1),
            }
        )

        stt_started = perf_counter()
        transcript = await transcribe(
            pcm_to_wav(utterance, start.sample_rate),
            "utterance.wav",
            "audio/wav",
            config.stt,
            client,
        )
        stt_ms = (perf_counter() - stt_started) * 1000
        await websocket.send_json(
            {
                "type": "transcript.completed",
                "text": transcript,
                "model": config.stt.model,
                "timing_ms": round(stt_ms, 1),
            }
        )

        await websocket.send_json(
            {
                "type": "response.started",
                "backend": backend_name,
                "model": llm_server.model,
            }
        )
        llm_started = perf_counter()
        first_token_ms = None
        answer_parts: list[str] = []
        async for delta in stream_completion(transcript, llm_server, client):
            if first_token_ms is None:
                first_token_ms = (perf_counter() - llm_started) * 1000
            answer_parts.append(delta)
            await websocket.send_json({"type": "response.text.delta", "delta": delta})

        llm_ms = (perf_counter() - llm_started) * 1000
        total_ms = (perf_counter() - processing_started) * 1000
        await websocket.send_json(
            {
                "type": "response.completed",
                "text": "".join(answer_parts),
                "timings_ms": {
                    "vad": round(audio.vad_time_ms, 1),
                    "stt": round(stt_ms, 1),
                    "llm_first_token": (
                        round(first_token_ms, 1)
                        if first_token_ms is not None
                        else None
                    ),
                    "llm_total": round(llm_ms, 1),
                    "processing_total": round(total_ms, 1),
                    "audio_duration": round(audio_duration_ms, 1),
                },
            }
        )
    except WebSocketDisconnect:
        return
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        await send_error(websocket, "protocol", str(exc))
    except ProviderError as exc:
        stage = "stt" if "STT" in str(exc) else "llm"
        await send_error(websocket, stage, str(exc))
