"""Microsoft TTS."""

import asyncio
import html
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import azure.cognitiveservices.speech as speechsdk

from .download import get_voices

_LOGGER = logging.getLogger(__name__)

# Raw PCM avoids a WAV header in the stream so we can hand bytes straight
# to the Wyoming client with a fixed, known format.
_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1


@dataclass
class _CallState:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    error: list[Exception] = field(default_factory=list)


class MicrosoftTTS:
    """Class to handle Microsoft TTS."""

    def __init__(self, args) -> None:
        """Initialize."""
        _LOGGER.debug("Initialize Microsoft TTS")
        self.args = args
        self.speech_config = speechsdk.SpeechConfig(
            subscription=args.subscription_key, region=args.service_region
        )
        self.speech_config.set_speech_synthesis_output_format(_OUTPUT_FORMAT)
        self.voices = get_voices(args.download_dir)

        # One synthesizer per instance — its WebSocket is reused across
        # calls, cutting per-sentence TTFB from ~600ms to ~90ms.
        # synthesize_stream is single-call at a time per instance.
        self._synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )
        self._call: _CallState | None = None
        self._synthesizer.synthesizing.connect(self._on_synthesizing)
        self._synthesizer.synthesis_completed.connect(self._on_completed)
        self._synthesizer.synthesis_canceled.connect(self._on_canceled)

    def _on_synthesizing(self, evt) -> None:
        call = self._call
        if call is None:
            return
        data = evt.result.audio_data
        if data:
            call.loop.call_soon_threadsafe(call.queue.put_nowait, data)

    def _on_completed(self, evt) -> None:
        call = self._call
        if call is None:
            return
        call.loop.call_soon_threadsafe(call.queue.put_nowait, None)

    def _on_canceled(self, evt) -> None:
        call = self._call
        if call is None:
            return
        details = evt.result.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error:
            call.error.append(
                RuntimeError(f"Azure TTS canceled: {details.error_details}")
            )
        call.loop.call_soon_threadsafe(call.queue.put_nowait, None)

    def _build_ssml(self, text, voice):
        """Build SSML embedding the voice (and any prosody/style flags).

        Voice must be set via SSML — mutating speech_config.speech_synthesis_voice_name
        is a no-op once the synthesizer has been constructed.
        """
        voice_key = self.voices[voice]["key"]
        voice_lang = self.voices[voice]["language"]["code"]
        safe_text = html.escape(text, quote=False)

        ssml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"',
        ]

        if self.args.style or self.args.style_degree:
            ssml_parts.append(' xmlns:mstts="https://www.w3.org/2001/mstts"')

        ssml_parts.append(f' xml:lang="{voice_lang}">')
        ssml_parts.append(f'<voice name="{voice_key}">')

        has_style = self.args.style is not None
        has_prosody = any([self.args.rate, self.args.pitch, self.args.volume])

        if has_style:
            style_attrs = [f'style="{self.args.style}"']
            if self.args.style_degree is not None:
                style_attrs.append(f'styledegree="{self.args.style_degree}"')
            ssml_parts.append(f'<mstts:express-as {" ".join(style_attrs)}>')

        if has_prosody:
            prosody_attrs = []
            if self.args.rate:
                prosody_attrs.append(f'rate="{self.args.rate}"')
            if self.args.pitch:
                prosody_attrs.append(f'pitch="{self.args.pitch}"')
            if self.args.volume:
                prosody_attrs.append(f'volume="{self.args.volume}"')
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')

        ssml_parts.append(safe_text)

        if has_prosody:
            ssml_parts.append('</prosody>')

        if has_style:
            ssml_parts.append('</mstts:express-as>')

        ssml_parts.append('</voice>')
        ssml_parts.append('</speak>')

        return ''.join(ssml_parts)

    async def synthesize_stream(
        self, text: str, voice: str | None = None
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM bytes as Azure produces them.

        Format is fixed by _OUTPUT_FORMAT (24 kHz, 16-bit, mono).
        """
        _LOGGER.debug("Requested TTS for [%s]", text)
        if voice is None:
            voice = self.args.voice

        loop = asyncio.get_running_loop()
        self._call = _CallState(loop=loop, queue=asyncio.Queue())

        # Always synthesize via SSML so the voice is set per-request — the
        # cached synthesizer ignores mutations to speech_config.speech_synthesis_voice_name.
        ssml = self._build_ssml(text, voice)
        _LOGGER.debug("Using SSML: %s", ssml)
        future = self._synthesizer.start_speaking_ssml_async(ssml)

        try:
            while True:
                chunk = await self._call.queue.get()
                if chunk is None:
                    break
                yield chunk
            if self._call.error:
                raise self._call.error[0]
        finally:
            self._call = None
            # Reap the SDK future so its result is consumed.
            await loop.run_in_executor(None, future.get)

    async def warmup(self) -> None:
        """Run a tiny synth so the first real request doesn't pay TLS/auth
        setup latency. SDK-internal state (auth tokens, etc.) appears to be
        cached globally, so warming one instance reduces TTFB on other
        fresh instances as well.
        """
        try:
            async for _ in self.synthesize_stream(
                "Warmup.", voice=self.args.voice
            ):
                pass
            _LOGGER.info("Azure TTS warmed up")
        except Exception as e:
            _LOGGER.warning("Azure TTS warmup failed: %s", e)
