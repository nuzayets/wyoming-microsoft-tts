"""Event handler for clients of the server."""

import argparse
import logging
import time

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .microsoft_tts import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, MicrosoftTTS
from .sentence_boundary import SentenceBoundaryDetector, remove_asterisks

_LOGGER = logging.getLogger(__name__)


class MicrosoftEventHandler(AsyncEventHandler):
    """Event handler for clients of the server."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        """Initialize."""
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.microsoft_tts = MicrosoftTTS(cli_args)
        self.sbd = SentenceBoundaryDetector(ssml=cli_args.ssml_input)
        self.is_streaming: bool | None = None
        self._synthesize: Synthesize | None = None
        # AudioStart is emitted once per stream (on the first chunk) and
        # AudioStop once at the end. HA's wyoming tts.py honors only the
        # first AudioStart and ignores per-sentence framing.
        self._stream_audio_started: bool = False
        # Per-stream prefill: hold the first cli_args.prefill_ms of audio
        # before emitting AudioStart, so the satellite has buffer headroom
        # to absorb mid-stream Azure pacing jitter. Drained on the first
        # chunk whose accumulated total meets the target, or by
        # _finish_stream if the utterance ends short.
        self._prefill_target_bytes: int = (
            cli_args.prefill_ms * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS // 1000
        )
        self._prefill_buffer: list[bytes] = []
        self._prefill_bytes: int = 0
        # Wall clock for debug timing logs: monotonic time when this stream
        # received SynthesizeStart (or when a legacy Synthesize started).
        self._stream_started_at: float | None = None
        # Per-stream emission accounting — counts every AudioChunk we send
        # so we can prove the full byte total left the server.
        self._stream_chunks_emitted: int = 0
        self._stream_bytes_emitted: int = 0
        self._last_emit_at: float | None = None

    async def handle_event(self, event: Event) -> bool:  # noqa: C901
        """Handle an event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                # Legacy non-streaming TTS request.
                if self.is_streaming:
                    return True
                synthesize = Synthesize.from_event(event)
                if not self.cli_args.ssml_input:
                    synthesize.text = remove_asterisks(synthesize.text)
                self._stream_audio_started = False
                self._reset_prefill()
                self._reset_emit_stats()
                self._stream_started_at = time.monotonic()
                _LOGGER.debug("[recv] Synthesize (legacy)")
                await self._handle_synthesize(synthesize)
                await self._finish_stream()
                self._stream_started_at = None
                return True

            if self.cli_args.no_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector(ssml=self.cli_args.ssml_input)
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                self._stream_audio_started = False
                self._reset_prefill()
                self._reset_emit_stats()
                self._stream_started_at = time.monotonic()
                _LOGGER.debug(
                    "[recv] SynthesizeStart voice=%s", stream_start.voice
                )
                return True

            if SynthesizeChunk.is_type(event.type):
                if not self.is_streaming or self._synthesize is None:
                    _LOGGER.warning(
                        "Got SynthesizeChunk outside an active stream"
                    )
                    return True
                stream_chunk = SynthesizeChunk.from_event(event)
                _LOGGER.debug(
                    "[recv] SynthesizeChunk len=%d t+%.3fs",
                    len(stream_chunk.text),
                    self._since_stream_start(),
                )
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug(
                        "[sbd] yielded at t+%.3fs: %s",
                        self._since_stream_start(),
                        sentence,
                    )
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)
                return True

            if SynthesizeStop.is_type(event.type):
                _LOGGER.debug(
                    "[recv] SynthesizeStop t+%.3fs",
                    self._since_stream_start(),
                )
                if self.is_streaming and self._synthesize is not None:
                    self._synthesize.text = self.sbd.finish()
                    if self._synthesize.text:
                        _LOGGER.debug(
                            "[sbd] flushing final at t+%.3fs: %s",
                            self._since_stream_start(),
                            self._synthesize.text,
                        )
                        await self._handle_synthesize(self._synthesize)
                await self._finish_stream()
                await self.write_event(SynthesizeStopped().event())
                _LOGGER.debug(
                    "[send] SynthesizeStopped t+%.3fs",
                    self._since_stream_start(),
                )
                self.is_streaming = False
                self._synthesize = None
                self.sbd = SentenceBoundaryDetector(ssml=self.cli_args.ssml_input)
                self._stream_started_at = None
                return True

            return True
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(  # noqa: C901
        self, synthesize: Synthesize
    ) -> None:
        """Synthesize one sentence and stream chunks to the client.

        Emits AudioStart on the very first chunk of the *stream* (not of this
        call). The caller emits AudioStop once when the stream ends via
        ``_finish_stream``.
        """
        raw_text = synthesize.text
        text = " ".join(raw_text.strip().splitlines())

        if synthesize.voice is None:
            voice = self.cli_args.voice
        else:
            voice = synthesize.voice.name

        if (
            self.cli_args.auto_punctuation
            and text
            and not self.cli_args.ssml_input
        ):
            has_punctuation = any(
                text[-1] == p for p in self.cli_args.auto_punctuation
            )
            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        _LOGGER.debug("Synthesizing: %s", text)
        rate, width, channels = SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS
        bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
        buf = b""
        first_chunk_logged = False

        async def emit_chunk(chunk_bytes: bytes) -> None:
            nonlocal first_chunk_logged
            # While the stream hasn't started, hold chunks in the prefill
            # buffer. Start the stream (AudioStart + drain) once the
            # buffered total meets the prefill target, or immediately if
            # prefill is disabled (target == 0).
            if not self._stream_audio_started:
                self._prefill_buffer.append(chunk_bytes)
                self._prefill_bytes += len(chunk_bytes)
                if self._prefill_bytes < self._prefill_target_bytes:
                    return
                await self._start_stream_audio(rate, width, channels)
                if not first_chunk_logged:
                    _LOGGER.debug(
                        "[send] first AudioChunk t+%.3fs (bytes=%d)",
                        self._since_stream_start(),
                        len(self._prefill_buffer[0]),
                    )
                    first_chunk_logged = True
                for held in self._prefill_buffer:
                    await self._write_audio_chunk(held, rate, width, channels)
                self._prefill_buffer.clear()
                self._prefill_bytes = 0
                return

            await self._write_audio_chunk(chunk_bytes, rate, width, channels)
            if not first_chunk_logged:
                _LOGGER.debug(
                    "[send] first AudioChunk t+%.3fs (bytes=%d)",
                    self._since_stream_start(),
                    len(chunk_bytes),
                )
                first_chunk_logged = True

        try:
            async for raw in self.microsoft_tts.synthesize_stream(
                text=text, voice=voice
            ):
                buf += raw
                while len(buf) >= bytes_per_chunk:
                    chunk, buf = buf[:bytes_per_chunk], buf[bytes_per_chunk:]
                    await emit_chunk(chunk)
            if buf:
                await emit_chunk(buf)
        except Exception as e:
            _LOGGER.error("Failed to synthesize/stream audio: %s", e)
            raise

    async def _finish_stream(self) -> None:
        """Emit AudioStop if any audio was sent for the current stream.

        If the utterance ended before the prefill target was reached, drain
        whatever's still buffered so the satellite hears the full audio
        rather than losing the tail.
        """
        if self._prefill_buffer:
            await self._start_stream_audio(SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS)
            for held in self._prefill_buffer:
                await self._write_audio_chunk(
                    held, SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS
                )
            _LOGGER.debug(
                "[send] prefill flushed (short utterance, %d bytes) t+%.3fs",
                self._prefill_bytes,
                self._since_stream_start(),
            )
            self._prefill_buffer.clear()
            self._prefill_bytes = 0

        if self._stream_audio_started:
            await self.write_event(AudioStop().event())
            _LOGGER.info(
                "[stream-end] chunks=%d bytes=%d (%.2fs audio) t+%.3fs",
                self._stream_chunks_emitted,
                self._stream_bytes_emitted,
                self._stream_bytes_emitted
                / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS),
                self._since_stream_start(),
            )
            _LOGGER.debug(
                "[send] AudioStop t+%.3fs", self._since_stream_start()
            )
            self._stream_audio_started = False

    async def _start_stream_audio(
        self, rate: int, width: int, channels: int
    ) -> None:
        """Emit AudioStart once per stream and log the prefill that filled it."""
        if self._stream_audio_started:
            return
        await self.write_event(
            AudioStart(rate=rate, width=width, channels=channels).event(),
        )
        self._stream_audio_started = True
        _LOGGER.debug(
            "[send] AudioStart t+%.3fs (prefill=%d/%d bytes)",
            self._since_stream_start(),
            self._prefill_bytes,
            self._prefill_target_bytes,
        )

    def _reset_prefill(self) -> None:
        """Drop any prefill state so the next stream starts from a clean slate."""
        self._prefill_buffer.clear()
        self._prefill_bytes = 0

    def _reset_emit_stats(self) -> None:
        """Reset per-stream emission counters."""
        self._stream_chunks_emitted = 0
        self._stream_bytes_emitted = 0
        self._last_emit_at = None

    async def _write_audio_chunk(
        self, audio: bytes, rate: int, width: int, channels: int
    ) -> None:
        """Send one AudioChunk and update per-stream accounting.

        Logs the chunk index, size, cumulative bytes, and time since the
        previous emit so we can spot gaps between consecutive chunks
        leaving the server.
        """
        before = time.monotonic()
        gap_ms = (
            int((before - self._last_emit_at) * 1000)
            if self._last_emit_at is not None
            else 0
        )
        await self.write_event(
            AudioChunk(
                audio=audio, rate=rate, width=width, channels=channels
            ).event(),
        )
        after = time.monotonic()
        self._stream_chunks_emitted += 1
        self._stream_bytes_emitted += len(audio)
        self._last_emit_at = after
        _LOGGER.debug(
            "[send] chunk #%d size=%d total=%d t+%.3fs gap=%dms write=%dms",
            self._stream_chunks_emitted,
            len(audio),
            self._stream_bytes_emitted,
            self._since_stream_start(),
            gap_ms,
            int((after - before) * 1000),
        )

    def _since_stream_start(self) -> float:
        """Seconds since this stream began; used only for debug log relative times."""
        if self._stream_started_at is None:
            return 0.0
        return time.monotonic() - self._stream_started_at
