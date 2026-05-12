"""Tests for the MicrosoftTTS class."""

from types import SimpleNamespace
import os
import pytest
from wyoming_microsoft_tts.microsoft_tts import MicrosoftTTS


def test_initialize(microsoft_tts, configuration):
    """Test initialization."""
    assert microsoft_tts.args.voice == configuration["voice"]
    assert microsoft_tts.speech_config is not None


@pytest.mark.skipif(
    not os.environ.get("SPEECH_KEY") or not os.environ.get("SPEECH_REGION"),
    reason="SPEECH_KEY and SPEECH_REGION environment variables required",
)
@pytest.mark.asyncio
async def test_synthesize_stream(microsoft_tts):
    """Test synthesize_stream produces raw PCM bytes."""
    chunks = [c async for c in microsoft_tts.synthesize_stream(
        "Hello, world!", "en-US-JennyNeural"
    )]
    assert chunks
    assert all(isinstance(c, (bytes, bytearray)) for c in chunks)
    assert sum(len(c) for c in chunks) > 0


# SSML Building Tests


def _ssml_args(**overrides):
    """Build a SimpleNamespace matching what MicrosoftTTS reads from args."""
    base = {
        "subscription_key": os.environ.get("SPEECH_KEY"),
        "service_region": os.environ.get("SPEECH_REGION"),
        "download_dir": "/tmp/",
        "voice": "en-US-JennyNeural",
        "rate": None,
        "pitch": None,
        "volume": None,
        "style": None,
        "style_degree": None,
        "ssml_input": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_ssml_with_rate():
    """Test SSML generation with rate parameter."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate="+30%",
        pitch=None,
        volume=None,
        style=None,
        style_degree=None,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("Hello, world!", "en-US-JennyNeural")

    assert '<?xml version="1.0" encoding="UTF-8"?>' in ssml
    assert '<speak version="1.0"' in ssml
    assert '<prosody rate="+30%">' in ssml
    assert "</prosody>" in ssml
    assert "Hello, world!" in ssml
    assert "xmlns:mstts" not in ssml  # No style, so no mstts namespace


def test_build_ssml_with_pitch():
    """Test SSML generation with pitch parameter."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate=None,
        pitch="+10%",
        volume=None,
        style=None,
        style_degree=None,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("Testing pitch", "en-US-JennyNeural")

    assert '<prosody pitch="+10%">' in ssml
    assert "</prosody>" in ssml
    assert "Testing pitch" in ssml


def test_build_ssml_with_volume():
    """Test SSML generation with volume parameter."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate=None,
        pitch=None,
        volume="loud",
        style=None,
        style_degree=None,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("Volume test", "en-US-JennyNeural")

    assert '<prosody volume="loud">' in ssml
    assert "</prosody>" in ssml
    assert "Volume test" in ssml


def test_build_ssml_with_all_prosody():
    """Test SSML generation with all prosody parameters."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate="fast",
        pitch="high",
        volume="+20%",
        style=None,
        style_degree=None,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("All prosody", "en-US-JennyNeural")

    assert '<prosody rate="fast" pitch="high" volume="+20%">' in ssml
    assert "</prosody>" in ssml
    assert "All prosody" in ssml


def test_build_ssml_with_style():
    """Test SSML generation with style parameter."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate=None,
        pitch=None,
        volume=None,
        style="cheerful",
        style_degree=None,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("Style test", "en-US-JennyNeural")

    assert 'xmlns:mstts="https://www.w3.org/2001/mstts"' in ssml
    assert '<mstts:express-as style="cheerful">' in ssml
    assert "</mstts:express-as>" in ssml
    assert "Style test" in ssml


def test_build_ssml_with_style_and_degree():
    """Test SSML generation with style and style_degree parameters."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate=None,
        pitch=None,
        volume=None,
        style="sad",
        style_degree=1.5,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("Sad voice", "en-US-JennyNeural")

    assert 'xmlns:mstts="https://www.w3.org/2001/mstts"' in ssml
    assert '<mstts:express-as style="sad" styledegree="1.5">' in ssml
    assert "</mstts:express-as>" in ssml
    assert "Sad voice" in ssml


def test_build_ssml_with_prosody_and_style():
    """Test SSML generation with both prosody and style parameters."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-US-JennyNeural",
        rate="slow",
        pitch="low",
        volume="soft",
        style="calm",
        style_degree=0.5,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("Combined test", "en-US-JennyNeural")

    assert 'xmlns:mstts="https://www.w3.org/2001/mstts"' in ssml
    assert '<mstts:express-as style="calm" styledegree="0.5">' in ssml
    assert '<prosody rate="slow" pitch="low" volume="soft">' in ssml
    assert "</prosody>" in ssml
    assert "</mstts:express-as>" in ssml
    assert "Combined test" in ssml


def test_build_ssml_voice_key_and_lang():
    """Test that SSML uses correct voice key and language."""
    args = SimpleNamespace(
        subscription_key=os.environ.get("SPEECH_KEY"),
        service_region=os.environ.get("SPEECH_REGION"),
        download_dir="/tmp/",
        voice="en-GB-SoniaNeural",
        rate="+10%",
        pitch=None,
        volume=None,
        style=None,
        style_degree=None,
    )
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml("UK voice", "en-GB-SoniaNeural")

    # Should contain the voice key from the voices.json
    assert 'xml:lang="en-GB"' in ssml
    assert '<voice name="en-GB-SoniaNeural">' in ssml


# SSML-input mode (model emits SSML fragments)


def test_build_ssml_input_fragment_not_escaped():
    """Inner SSML tags are preserved (not html-escaped) in SSML-input mode."""
    args = _ssml_args(ssml_input=True)
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml(
        "<prosody rate='slow'>Hello.</prosody>", "en-US-JennyNeural"
    )
    assert "<prosody rate='slow'>Hello.</prosody>" in ssml
    # Must NOT appear escaped:
    assert "&lt;prosody" not in ssml


def test_build_ssml_input_always_emits_mstts_namespace():
    """The mstts namespace is always declared so <mstts:express-as> works."""
    args = _ssml_args(ssml_input=True)
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml(
        "<mstts:express-as style='cheerful'>Hi.</mstts:express-as>",
        "en-US-JennyNeural",
    )
    assert 'xmlns:mstts="https://www.w3.org/2001/mstts"' in ssml
    assert "<mstts:express-as style='cheerful'>Hi.</mstts:express-as>" in ssml


def test_build_ssml_input_strips_model_speak_envelope():
    """A full <speak> document from the model is reduced to its inner content."""
    args = _ssml_args(ssml_input=True)
    tts = MicrosoftTTS(args)
    fragment = (
        '<?xml version="1.0"?>'
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US">'
        '<voice name="en-US-AriaNeural">'
        '<prosody rate="slow">Hello.</prosody>'
        '</voice></speak>'
    )
    ssml = tts._build_ssml(fragment, "en-US-JennyNeural")
    # Operator's voice wins:
    assert '<voice name="en-US-JennyNeural">' in ssml
    assert "AriaNeural" not in ssml
    # Inner content survives:
    assert '<prosody rate="slow">Hello.</prosody>' in ssml
    # No double envelopes:
    assert ssml.count("<speak") == 1
    assert ssml.count("</speak>") == 1
    assert ssml.count("<voice") == 1
    assert ssml.count("</voice>") == 1


def test_build_ssml_input_pins_operator_voice_and_lang():
    """Operator voice and xml:lang come from CLI args, not the model."""
    args = _ssml_args(ssml_input=True, voice="en-GB-SoniaNeural")
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml(
        '<voice name="en-US-AriaNeural">Hi.</voice>', "en-GB-SoniaNeural"
    )
    assert '<voice name="en-GB-SoniaNeural">' in ssml
    assert 'xml:lang="en-GB"' in ssml
    assert "AriaNeural" not in ssml


def test_build_ssml_input_preserves_operator_prosody_wrapper():
    """Operator's --rate still wraps the model's content."""
    args = _ssml_args(ssml_input=True, rate="+30%")
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml(
        "<emphasis>Hi.</emphasis>", "en-US-JennyNeural"
    )
    assert '<prosody rate="+30%">' in ssml
    assert "<emphasis>Hi.</emphasis>" in ssml
    # Operator's prosody must wrap (open before, close after) the inner content.
    prosody_open = ssml.index('<prosody rate="+30%">')
    inner = ssml.index("<emphasis>")
    prosody_close = ssml.index("</prosody>")
    assert prosody_open < inner < prosody_close


def test_build_ssml_input_off_escapes_angle_brackets():
    """With ssml_input disabled, raw < and > must still be escaped (regression)."""
    args = _ssml_args(ssml_input=False)
    tts = MicrosoftTTS(args)
    ssml = tts._build_ssml(
        "<prosody>Hello.</prosody>", "en-US-JennyNeural"
    )
    # Treated as literal text → escaped:
    assert "&lt;prosody&gt;" in ssml
    assert "<prosody>Hello.</prosody>" not in ssml


# Integration Tests with synthesize_stream


@pytest.mark.skipif(
    not os.environ.get("SPEECH_KEY") or not os.environ.get("SPEECH_REGION"),
    reason="SPEECH_KEY and SPEECH_REGION environment variables required",
)
@pytest.mark.parametrize(
    "extra_args",
    [
        {"rate": "+30%"},
        {"pitch": "+5%"},
        {"volume": "loud"},
        {"style": "cheerful"},
        {"rate": "fast", "pitch": "+10%", "volume": "loud",
         "style": "excited", "style_degree": 1.2},
        {},
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_with_params(extra_args):
    """synthesize_stream produces bytes for various SSML param combinations."""
    base = {
        "subscription_key": os.environ.get("SPEECH_KEY"),
        "service_region": os.environ.get("SPEECH_REGION"),
        "download_dir": "/tmp/",
        "voice": "en-US-JennyNeural",
        "rate": None,
        "pitch": None,
        "volume": None,
        "style": None,
        "style_degree": None,
    }
    base.update(extra_args)
    tts = MicrosoftTTS(SimpleNamespace(**base))
    total = 0
    async for chunk in tts.synthesize_stream(
        "Testing parameters", "en-US-JennyNeural"
    ):
        total += len(chunk)
    assert total > 0
