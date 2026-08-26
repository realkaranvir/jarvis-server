import argparse
import asyncio
import json
import wave

from websockets.asyncio.client import connect


async def run(url: str, audio_path: str) -> None:
    with wave.open(audio_path, "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Audio must be mono 16-bit PCM WAV")
        sample_rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())

    async with connect(url) as websocket:
        print(await websocket.recv())
        await websocket.send(
            json.dumps({"type": "session.start", "sample_rate": sample_rate})
        )
        print(await websocket.recv())

        chunk_bytes = sample_rate // 10 * 2
        for offset in range(0, len(frames), chunk_bytes):
            await websocket.send(frames[offset : offset + chunk_bytes])
        await websocket.send(b"\x00\x00" * sample_rate)

        async for raw_event in websocket:
            print(raw_event)
            if json.loads(raw_event)["type"] in {
                "response.completed",
                "error",
            }:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Mono 16-bit PCM WAV file")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/conversation",
    )
    arguments = parser.parse_args()
    asyncio.run(run(arguments.url, arguments.audio))


if __name__ == "__main__":
    main()
