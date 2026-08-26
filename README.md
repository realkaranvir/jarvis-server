# Jarvis Server

FastAPI server for the Jarvis audio/LLM pipeline. The `/llm` and `/stt`
endpoints forward requests to provider services selected in configuration.

## Setup

The machine-local `config.json` is ignored by Git. Start from the committed
example when setting up another machine:

```shell
cp config.example.json config.json
```

Each entry in `llm.servers` defines a selectable LLM backend. The
`llm.default` setting controls which backend is used when a request omits
`backend`. The `stt` section selects an OpenAI-compatible transcription
provider.
Set `JARVIS_CONFIG` to use a config file at a different path.

Run the development server:

```shell
uv run fastapi dev
```

Send a prompt to the default backend:

```shell
curl http://127.0.0.1:8000/llm \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello in one sentence."}'
```

Select the smaller vision-capable model (text requests only for now):

```shell
curl http://127.0.0.1:8000/llm \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello briefly.","backend":"vision"}'
```

Transcribe an audio file:

```shell
curl http://127.0.0.1:8000/stt \
  -F 'audio=@sample.wav'
```

## Streaming conversation

Connect to `ws://127.0.0.1:8000/conversation`, then send:

1. `{"type":"session.start","sample_rate":16000}` as JSON.
2. Mono PCM16 little-endian audio as binary WebSocket messages.
3. Optionally `{"type":"audio.commit"}` to end the turn manually; otherwise
   server-side VAD ends it after the configured silence interval.

The server emits `transcript.completed`, `response.text.delta`, and
`response.completed` events. Completion includes VAD, STT, LLM first-token,
LLM total, total processing, and audio-duration timings in milliseconds.

Test with an existing mono PCM16 WAV file:

```shell
uv run python examples/conversation_client.py sample.wav
```
