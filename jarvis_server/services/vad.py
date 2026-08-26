from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import webrtcvad

from jarvis_server.config import VADConfig


SUPPORTED_SAMPLE_RATES = {8000, 16000, 32000, 48000}


class VADDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


@dataclass(frozen=True)
class VADEvent:
    type: str
    audio: bytes | None = None
    reason: str | None = None


class UtteranceBuffer:
    def __init__(
        self,
        sample_rate: int,
        config: VADConfig,
        detector: VADDetector | None = None,
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        self.sample_rate = sample_rate
        self.config = config
        self.frame_bytes = sample_rate * config.frame_duration_ms // 1000 * 2
        self.detector = detector or webrtcvad.Vad(config.mode)
        self.pending = bytearray()
        self.received = bytearray()
        self.utterance = bytearray()
        self.pre_roll: deque[bytes] = deque(
            maxlen=max(1, config.prefix_padding_ms // config.frame_duration_ms)
        )
        self.speech_active = False
        self.consecutive_voiced = 0
        self.consecutive_silent = 0
        self.vad_time_ms = 0.0

    def feed(self, data: bytes) -> list[VADEvent]:
        self.pending.extend(data)
        self.received.extend(data)
        max_bytes = self.sample_rate * 2 * self.config.max_speech_duration_seconds
        if len(self.received) > max_bytes:
            del self.received[:-max_bytes]

        events: list[VADEvent] = []
        while len(self.pending) >= self.frame_bytes:
            frame = bytes(self.pending[: self.frame_bytes])
            del self.pending[: self.frame_bytes]
            events.extend(self._process_frame(frame))
        return events

    def commit(self) -> bytes | None:
        audio = bytes(self.received)
        self._reset()
        return audio or None

    def _process_frame(self, frame: bytes) -> list[VADEvent]:
        started = perf_counter()
        voiced = self.detector.is_speech(frame, self.sample_rate)
        self.vad_time_ms += (perf_counter() - started) * 1000

        if not self.speech_active:
            self.pre_roll.append(frame)
            self.consecutive_voiced = self.consecutive_voiced + 1 if voiced else 0
            if self.consecutive_voiced < 3:
                return []

            self.speech_active = True
            self.utterance.extend(b"".join(self.pre_roll))
            self.consecutive_silent = 0
            return [VADEvent(type="speech.started")]

        self.utterance.extend(frame)
        self.consecutive_silent = 0 if voiced else self.consecutive_silent + 1

        silence_frames = max(
            1,
            self.config.silence_duration_ms // self.config.frame_duration_ms,
        )
        min_speech_bytes = (
            self.sample_rate * 2 * self.config.min_speech_duration_ms // 1000
        )
        max_speech_bytes = (
            self.sample_rate * 2 * self.config.max_speech_duration_seconds
        )

        reason = None
        if (
            self.consecutive_silent >= silence_frames
            and len(self.utterance) >= min_speech_bytes
        ):
            reason = "silence"
        elif len(self.utterance) >= max_speech_bytes:
            reason = "max_duration"
        if reason is None:
            return []

        audio = bytes(self.utterance)
        self._reset()
        return [VADEvent(type="speech.stopped", audio=audio, reason=reason)]

    def _reset(self) -> None:
        self.pending.clear()
        self.received.clear()
        self.utterance.clear()
        self.pre_roll.clear()
        self.speech_active = False
        self.consecutive_voiced = 0
        self.consecutive_silent = 0
