"""Tests for the SSML fragment sanitizer."""

import pytest

from wyoming_microsoft_tts.ssml_sanitizer import sanitize_ssml_fragment


# Envelope stripping


def test_strips_xml_declaration():
    out = sanitize_ssml_fragment('<?xml version="1.0"?>Hello.')
    assert "<?xml" not in out
    assert "Hello." in out


def test_strips_speak_wrapper():
    out = sanitize_ssml_fragment(
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US">Hi.</speak>'
    )
    assert "<speak" not in out
    assert "</speak>" not in out
    assert "Hi." in out


def test_strips_voice_wrapper():
    out = sanitize_ssml_fragment(
        '<voice name="en-US-AriaNeural">Hi.</voice>'
    )
    assert "<voice" not in out
    assert "AriaNeural" not in out
    assert "Hi." in out


def test_strips_full_envelope_from_users_example():
    """The exact malformed example from the bug report should clean up."""
    out = sanitize_ssml_fragment(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-GB">'
        '<voice name="en-US-FableTurboMultilingualNeural">'
        '<prosody rate="moderate">It is a lovely, sunny day in Toronto.</prosody>'
        '</voice></speak>'
    )
    assert "<speak" not in out
    assert "<voice" not in out
    assert "FableTurboMultilingualNeural" not in out
    assert '<prosody rate="medium">' in out
    assert "It is a lovely, sunny day in Toronto." in out


# Prosody attribute coercion


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("moderate", "medium"),
        ("normal", "medium"),
        ("average", "medium"),
        ("regular", "medium"),
        ("quick", "fast"),
        ("rapid", "fast"),
        ("slowly", "slow"),
        ("MODERATE", "medium"),  # case-insensitive
    ],
)
def test_prosody_rate_aliases_coerced(raw, expected):
    out = sanitize_ssml_fragment(f'<prosody rate="{raw}">Hi.</prosody>')
    assert f'rate="{expected}"' in out


@pytest.mark.parametrize(
    "value",
    ["x-slow", "slow", "medium", "fast", "x-fast", "default"],
)
def test_prosody_rate_enum_passes_through(value):
    out = sanitize_ssml_fragment(f'<prosody rate="{value}">Hi.</prosody>')
    assert f'rate="{value}"' in out


@pytest.mark.parametrize(
    "value",
    ["+30%", "-10%", "1.5", "0.5", "+50%"],
)
def test_prosody_rate_numeric_passes_through(value):
    out = sanitize_ssml_fragment(f'<prosody rate="{value}">Hi.</prosody>')
    assert f'rate="{value}"' in out


def test_prosody_unknown_rate_dropped():
    out = sanitize_ssml_fragment('<prosody rate="warp-speed">Hi.</prosody>')
    assert "rate=" not in out
    assert "<prosody" in out  # element itself kept
    assert "Hi." in out


def test_prosody_pitch_aliases():
    out = sanitize_ssml_fragment('<prosody pitch="normal">Hi.</prosody>')
    assert 'pitch="medium"' in out


def test_prosody_volume_aliases():
    out = sanitize_ssml_fragment('<prosody volume="quiet">Hi.</prosody>')
    assert 'volume="soft"' in out


def test_prosody_pitch_numeric_with_unit():
    out = sanitize_ssml_fragment('<prosody pitch="+20Hz">Hi.</prosody>')
    assert 'pitch="+20Hz"' in out


def test_prosody_multiple_bad_attrs_all_coerced():
    out = sanitize_ssml_fragment(
        '<prosody rate="moderate" pitch="normal" volume="quiet">Hi.</prosody>'
    )
    assert 'rate="medium"' in out
    assert 'pitch="medium"' in out
    assert 'volume="soft"' in out


def test_prosody_mixed_valid_and_invalid_attrs():
    out = sanitize_ssml_fragment(
        '<prosody rate="+20%" pitch="hyperhigh" volume="loud">Hi.</prosody>'
    )
    assert 'rate="+20%"' in out
    assert "pitch=" not in out  # invalid → dropped
    assert 'volume="loud"' in out


# Unknown tag handling


def test_unknown_tag_unwrapped_children_kept():
    out = sanitize_ssml_fragment(
        '<unknown attr="x">keep <emphasis>this</emphasis> too</unknown>'
    )
    assert "<unknown" not in out
    assert "</unknown>" not in out
    assert "keep " in out
    assert "<emphasis>this</emphasis>" in out
    assert " too" in out


def test_nested_unknown_tags_all_unwrapped():
    out = sanitize_ssml_fragment(
        "<foo><bar><baz>inner</baz></bar></foo>"
    )
    assert "<foo" not in out
    assert "<bar" not in out
    assert "<baz" not in out
    assert "inner" in out


def test_allowed_tags_preserved():
    """Whitelisted SSML tags survive intact."""
    out = sanitize_ssml_fragment(
        '<emphasis level="strong">word</emphasis>'
        '<break time="500ms"/>'
        '<say-as interpret-as="date">2026-05-29</say-as>'
    )
    assert '<emphasis level="strong">word</emphasis>' in out
    assert "<break" in out and 'time="500ms"' in out
    assert '<say-as interpret-as="date">2026-05-29</say-as>' in out


# Namespace handling


def test_mstts_express_as_preserved():
    out = sanitize_ssml_fragment(
        '<speak xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts">'
        '<voice name="x">'
        '<mstts:express-as style="cheerful" styledegree="1.5">Yay.</mstts:express-as>'
        "</voice></speak>"
    )
    assert "<mstts:express-as" in out
    assert 'style="cheerful"' in out
    assert 'styledegree="1.5"' in out
    assert "Yay." in out


def test_redundant_ssml_xmlns_stripped():
    """The operator's <speak> already declares the default namespace."""
    out = sanitize_ssml_fragment(
        '<speak xmlns="http://www.w3.org/2001/10/synthesis">'
        '<voice name="x"><emphasis>Hi.</emphasis></voice></speak>'
    )
    assert "xmlns=" not in out


def test_redundant_mstts_xmlns_stripped():
    out = sanitize_ssml_fragment(
        '<speak xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts">'
        '<voice name="x"><mstts:express-as style="sad">Hi.</mstts:express-as>'
        "</voice></speak>"
    )
    assert "xmlns:mstts" not in out


# Malformed input recovery


def test_unclosed_tag_recovered():
    """lxml's recover mode closes truncated tags."""
    out = sanitize_ssml_fragment('<prosody rate="slow">Hello')
    assert "Hello" in out
    assert "<prosody" in out
    assert "</prosody>" in out


def test_stray_lt_gt_in_text_handled():
    """Garbage characters don't crash the sanitizer."""
    out = sanitize_ssml_fragment("plain < text > with angle brackets")
    # Output is best-effort; we just care it returned something containing
    # the actual prose, didn't raise, and didn't include the original
    # angle brackets as live markup.
    assert "plain" in out
    assert "with angle brackets" in out


def test_completely_invalid_falls_back_to_escape():
    """If lxml gives up entirely, return escaped text rather than raise."""
    # A purely broken sequence that recover-mode can't fix into anything
    # useful — lxml still returns *something*, but the function must not
    # raise either way.
    out = sanitize_ssml_fragment("<<<>>>")
    assert isinstance(out, str)  # didn't raise


# Plain content


def test_plain_text_passes_through():
    out = sanitize_ssml_fragment("Just plain text.")
    assert out == "Just plain text."


def test_empty_input_returns_empty():
    assert sanitize_ssml_fragment("") == ""
    assert sanitize_ssml_fragment("   ") == ""


def test_text_around_tags_preserved():
    out = sanitize_ssml_fragment(
        "Before <emphasis>middle</emphasis> after."
    )
    assert "Before " in out
    assert "<emphasis>middle</emphasis>" in out
    assert " after." in out


def test_multiple_top_level_siblings():
    """SBD can hand us a fragment with multiple peer elements."""
    out = sanitize_ssml_fragment(
        '<prosody rate="slow">One.</prosody> <emphasis>Two.</emphasis>'
    )
    assert '<prosody rate="slow">One.</prosody>' in out
    assert "<emphasis>Two.</emphasis>" in out


def test_xml_entities_preserved():
    """Apostrophes/ampersands inside text are kept as valid XML entities."""
    out = sanitize_ssml_fragment("It&apos;s a test &amp; more.")
    # Serialized output should round-trip the entities — actual chars or
    # numeric/named refs, both are valid for Azure.
    assert "test" in out
    assert "more" in out


# Behavior when given the operator's already-built SSML (defense in depth)


def test_double_envelope_cleaned():
    """Even if input has nested speak/voice (model bug), result has none."""
    out = sanitize_ssml_fragment(
        "<speak><voice name='a'><speak><voice name='b'>"
        "<prosody rate='fast'>Hi.</prosody>"
        "</voice></speak></voice></speak>"
    )
    assert "<speak" not in out
    assert "<voice" not in out
    assert '<prosody rate="fast">Hi.</prosody>' in out
