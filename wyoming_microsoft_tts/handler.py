"""Event handler for clients of the server."""

import argparse
import asyncio
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
        self.sbd = SentenceBoundaryDetector()
        self.is_streaming: bool | None = None
        self._synthesize: Synthesize | None = None
        # Pacing clock — shared across all sentences in a stream so playback
        # is continuous instead of resetting per sentence.
        self._pace_start: float | None = None
        self._pace_bytes_sent: int = 0
        # Wall clock for debug timing logs: monotonic time when this stream
        # received SynthesizeStart (or when a legacy Synthesize started).
        self._stream_started_at: float | None = None

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
                synthesize.text = remove_asterisks(synthesize.text)
                self._reset_pacing()
                self._stream_started_at = time.monotonic()
                _LOGGER.debug("[recv] Synthesize (legacy)")
                await self._handle_synthesize(synthesize, is_final=True)
                self._stream_started_at = None
                return True

            if self.cli_args.no_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                self._reset_pacing()
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
                        "[sbd] yielded sentence at t+%.3fs: %s",
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
                drained = False
                if self.is_streaming and self._synthesize is not None:
                    self._synthesize.text = self.sbd.finish()
                    if self._synthesize.text:
                        _LOGGER.debug(
                            "[sbd] flushing final at t+%.3fs: %s",
                            self._since_stream_start(),
                            self._synthesize.text,
                        )
                        await self._handle_synthesize(
                            self._synthesize, is_final=True
                        )
                        drained = True
                if self.is_streaming and not drained:
                    # No trailing sentence to synthesize, but earlier sentences
                    # left a pacing buffer in the client. Wait it out before
                    # signalling end-of-stream.
                    await self._pace_drain()
                await self.write_event(SynthesizeStopped().event())
                _LOGGER.debug(
                    "[send] SynthesizeStopped t+%.3fs",
                    self._since_stream_start(),
                )
                self.is_streaming = False
                self._synthesize = None
                self.sbd = SentenceBoundaryDetector()
                self._reset_pacing()
                self._stream_started_at = None
                return True

            return True
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(  # noqa: C901
        self, synthesize: Synthesize, *, is_final: bool = False
    ):
        _LOGGER.debug(synthesize)
        raw_text = synthesize.text

        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        if synthesize.voice is None:  # Use default voice if not specified
            voice = self.cli_args.voice
        else:
            voice = synthesize.voice.name

        if self.cli_args.auto_punctuation and text:
            # Add automatic punctuation (important for some voices)
            has_punctuation = False
            for punc_char in self.cli_args.auto_punctuation:
                if text[-1] == punc_char:
                    has_punctuation = True
                    break

            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        _LOGGER.debug("Synthesizing: %s", text)
        rate, width, channels = SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS
        bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
        audio_started = False
        buf = b""

        chunks_emitted = 0

        async def emit_chunk(chunk_bytes: bytes) -> None:
            nonlocal audio_started, chunks_emitted
            if not audio_started:
                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=channels).event(),
                )
                audio_started = True
                _LOGGER.debug(
                    "[send] AudioStart t+%.3fs", self._since_stream_start()
                )
            await self._pace_before_send(len(chunk_bytes))
            await self.write_event(
                AudioChunk(
                    audio=chunk_bytes,
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event(),
            )
            self._pace_bytes_sent += len(chunk_bytes)
            chunks_emitted += 1
            if chunks_emitted == 1:
                _LOGGER.debug(
                    "[send] first AudioChunk t+%.3fs (bytes=%d)",
                    self._since_stream_start(),
                    len(chunk_bytes),
                )

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
            if audio_started:
                await self.write_event(AudioStop().event())
            return False

        if audio_started:
            if is_final:
                await self._pace_drain()
            await self.write_event(AudioStop().event())
            _LOGGER.debug(
                "[send] AudioStop t+%.3fs (chunks=%d, is_final=%s)",
                self._since_stream_start(),
                chunks_emitted,
                is_final,
            )
        return True

    def _reset_pacing(self) -> None:
        """Reset the pacing clock at the start of a stream / legacy call."""
        self._pace_start = None
        self._pace_bytes_sent = 0

    def _since_stream_start(self) -> float:
        """Seconds since this stream began; used only for debug log relative times."""
        if self._stream_started_at is None:
            return 0.0
        return time.monotonic() - self._stream_started_at

    async def _pace_drain(self) -> None:
        """Wait for the paced audio to finish playing on the client.

        Pacing keeps the stream ``streaming_pacing_buffer_seconds`` ahead of
        real-time so the client doesn't underrun. At end of stream we sleep
        until that lead would have drained plus another buffer's worth, so
        ``AudioStop`` / ``SynthesizeStopped`` land *after* the satellite's
        speaker is actually empty. Without this the satellite transitions
        back to listening with audio still playing — the original Voice PE
        self-feedback loop (esphome/home-assistant-voice-pe#537).
        """
        if self.cli_args.no_streaming_pacing or self._pace_start is None:
            _LOGGER.debug(
                "[drain] skipped t+%.3fs (no_pacing=%s, pace_start=%s)",
                self._since_stream_start(),
                self.cli_args.no_streaming_pacing,
                self._pace_start,
            )
            return
        bytes_per_second = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
        audio_seconds_sent = self._pace_bytes_sent / bytes_per_second
        target_time = (
            self._pace_start
            + audio_seconds_sent
            + self.cli_args.streaming_pacing_buffer_seconds
        )
        sleep_for = target_time - time.monotonic()
        _LOGGER.debug(
            "[drain] t+%.3fs audio_sent=%.3fs sleep=%.3fs",
            self._since_stream_start(),
            audio_seconds_sent,
            max(0.0, sleep_for),
        )
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    async def _pace_before_send(self, next_chunk_bytes: int) -> None:
        """Hold a chunk back so the stream tracks real-time playback.

        We let the first ``--streaming-pacing-buffer-seconds`` of audio flow
        unthrottled to fill the client's playback buffer, then sleep before
        each subsequent chunk so the stream's end time approximates the
        speaker's playback end time. Fixes the Voice PE feedback loop where
        the satellite enters listening mode while audio is still playing
        (esphome/home-assistant-voice-pe#537).
        """
        if self.cli_args.no_streaming_pacing:
            return
        bytes_per_second = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
        now = time.monotonic()
        audio_seconds_sent = self._pace_bytes_sent / bytes_per_second
        # Re-anchor when wall time has pulled ahead of audio time, e.g. a
        # slow LLM stalled between sentence boundaries. Otherwise the
        # accumulated "debt" would let the next sentence burst out unpaced
        # and SynthesizeStopped would fire before the client finishes
        # playing it — the exact failure mode pacing exists to prevent.
        if (
            self._pace_start is None
            or now - self._pace_start > audio_seconds_sent
        ):
            self._pace_start = now - audio_seconds_sent
        target_time = (
            self._pace_start
            + audio_seconds_sent
            - self.cli_args.streaming_pacing_buffer_seconds
        )
        sleep_for = target_time - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
