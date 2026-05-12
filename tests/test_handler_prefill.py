"""Tests for the per-stream prefill buffer in MicrosoftEventHandler."""

import asyncio
import os
from types import SimpleNamespace

from wyoming.audio import AudioChunk, AudioStart
from wyoming.info import Info
from wyoming.tts import Synthesize

from wyoming_microsoft_tts.handler import MicrosoftEventHandler
from wyoming_microsoft_tts.microsoft_tts import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH


class _CapturingHandler(MicrosoftEventHandler):
    """Captures emitted events instead of writing them to a stream."""

    def __init__(self, *args, **kwargs):
        """Initialize and start with an empty event list."""
        super().__init__(*args, **kwargs)
        self.events: list = []

    async def write_event(self, event):  # type: ignore[override]
        """Append rather than write — lets tests inspect emission order."""
        self.events.append(event)


def _make_args(prefill_ms: int = 200, **overrides) -> SimpleNamespace:
    """Build a CLI-args namespace covering everything the handler reads."""
    base = {
        "subscription_key": os.environ.get("SPEECH_KEY", "test-fake-key"),
        "service_region": os.environ.get("SPEECH_REGION", "westus2"),
        "download_dir": "/tmp/",
        "voice": "en-US-JennyNeural",
        "rate": None,
        "pitch": None,
        "volume": None,
        "style": None,
        "style_degree": None,
        "ssml_input": False,
        "auto_punctuation": "",
        "samples_per_chunk": 1024,
        "no_streaming": False,
        "prefill_ms": prefill_ms,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_handler(prefill_ms: int = 200) -> _CapturingHandler:
    """Construct a handler whose synthesize_stream is replaced by tests."""
    args = _make_args(prefill_ms=prefill_ms)
    info = Info()
    return _CapturingHandler(info, args, None, None)  # type: ignore[arg-type]


def _fake_stream(raw_chunks: list[bytes]):
    """Build an async iterator that yields the given byte chunks."""

    async def gen(text, voice=None):
        for c in raw_chunks:
            yield c

    return gen


def _ms_to_bytes(ms: int) -> int:
    """Convert audio duration in ms to bytes at the wire format."""
    return (ms * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS) // 1000


def _audio_chunk_payload(event) -> bytes:
    """Extract raw PCM bytes from an AudioChunk event."""
    return AudioChunk.from_event(event).audio


def test_prefill_holds_audiostart_until_target_reached():
    """AudioStart must not fire before enough audio is buffered."""
    h = _make_handler(prefill_ms=100)
    target = _ms_to_bytes(100)  # 100 ms @ 24kHz 16-bit mono = 4800 bytes

    # 4 raw chunks of 1500 bytes each = 6000 bytes total. AudioStart should
    # fire only after the chunk that brings the total to >= target.
    raw = [b"a" * 1500, b"b" * 1500, b"c" * 1500, b"d" * 1500]
    h.microsoft_tts.synthesize_stream = _fake_stream(raw)

    asyncio.run(h._handle_synthesize(Synthesize(text="hi", voice=None)))
    asyncio.run(h._finish_stream())

    # Find the AudioStart and compute how much audio was already buffered
    # (= sum of AudioChunk payloads emitted BEFORE the AudioStart event,
    # which is none; chunks are drained AFTER AudioStart).
    types = [e.type for e in h.events]
    assert types[0] == "audio-start", f"AudioStart must be first; got {types}"

    # After AudioStart, the buffered chunks drain in order.
    chunk_events = [e for e in h.events if AudioChunk.is_type(e.type)]
    drained = b"".join(_audio_chunk_payload(e) for e in chunk_events)
    assert drained == b"".join(raw), "All audio must reach the wire"

    # The first event after AudioStart must already represent >= target bytes
    # being held — the drained buffer should have at least target bytes.
    first_drain_bytes = sum(
        len(_audio_chunk_payload(e))
        for e in h.events[1 : 1 + 4]
        if AudioChunk.is_type(e.type)
    )
    assert first_drain_bytes >= target


def test_prefill_short_utterance_is_flushed_on_finish():
    """An utterance shorter than the prefill target must still reach the satellite."""
    h = _make_handler(prefill_ms=500)  # 500 ms target = 24000 bytes
    raw = [b"x" * 1000]  # 1000 bytes, well below target
    h.microsoft_tts.synthesize_stream = _fake_stream(raw)

    asyncio.run(h._handle_synthesize(Synthesize(text="hi", voice=None)))

    # Mid-stream: AudioStart must NOT have fired yet (still below target).
    assert not any(AudioStart.is_type(e.type) for e in h.events)
    assert not any(AudioChunk.is_type(e.type) for e in h.events)

    # _finish_stream must drain the buffer and emit AudioStop.
    asyncio.run(h._finish_stream())

    types = [e.type for e in h.events]
    assert "audio-start" in types
    assert "audio-stop" in types

    chunks = [e for e in h.events if AudioChunk.is_type(e.type)]
    drained = b"".join(_audio_chunk_payload(e) for e in chunks)
    assert drained == b"x" * 1000


def test_prefill_disabled_emits_immediately():
    """prefill_ms=0 means no buffering — first chunk fires AudioStart."""
    h = _make_handler(prefill_ms=0)
    raw = [b"a" * 2048, b"b" * 2048]
    h.microsoft_tts.synthesize_stream = _fake_stream(raw)

    asyncio.run(h._handle_synthesize(Synthesize(text="hi", voice=None)))
    asyncio.run(h._finish_stream())

    # AudioStart fires before any AudioChunk.
    types = [e.type for e in h.events]
    audio_start_idx = types.index("audio-start")
    first_chunk_idx = next(
        i for i, e in enumerate(h.events) if AudioChunk.is_type(e.type)
    )
    assert audio_start_idx < first_chunk_idx


def test_prefill_subsequent_sentences_stream_immediately():
    """Only the first synthesize call in a stream pays the prefill cost."""
    h = _make_handler(prefill_ms=100)
    # First call: enough audio to cross prefill target on the first chunk.
    h.microsoft_tts.synthesize_stream = _fake_stream([b"a" * 8000])
    asyncio.run(h._handle_synthesize(Synthesize(text="first", voice=None)))

    audio_starts_after_first = sum(
        1 for e in h.events if AudioStart.is_type(e.type)
    )
    assert audio_starts_after_first == 1

    # Second call in the same stream: should NOT emit another AudioStart and
    # must NOT buffer — chunks flow immediately.
    h.events.clear()
    h.microsoft_tts.synthesize_stream = _fake_stream([b"b" * 200])
    asyncio.run(h._handle_synthesize(Synthesize(text="second", voice=None)))

    assert not any(AudioStart.is_type(e.type) for e in h.events)
    chunks = [e for e in h.events if AudioChunk.is_type(e.type)]
    assert chunks, "Second sentence chunks must flow without prefill"
    assert _audio_chunk_payload(chunks[0]) == b"b" * 200


def test_prefill_resets_between_streams():
    """A new stream must rebuild the prefill cushion from scratch."""
    h = _make_handler(prefill_ms=500)
    # Stream 1: short utterance that ends below target; flushed at finish.
    h.microsoft_tts.synthesize_stream = _fake_stream([b"x" * 1000])
    asyncio.run(h._handle_synthesize(Synthesize(text="one", voice=None)))
    asyncio.run(h._finish_stream())

    assert h._prefill_buffer == []
    assert h._prefill_bytes == 0
    assert h._stream_audio_started is False

    # Stream 2: must hold chunks again until target is reached.
    h.events.clear()
    h.microsoft_tts.synthesize_stream = _fake_stream([b"y" * 100])
    asyncio.run(h._handle_synthesize(Synthesize(text="two", voice=None)))
    # 100 bytes is well under the 24000-byte target → no AudioStart yet.
    assert not any(AudioStart.is_type(e.type) for e in h.events)
    assert h._prefill_bytes == 100
