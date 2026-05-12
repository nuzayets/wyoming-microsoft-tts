"""Tests for SentenceBoundaryDetector — plain and SSML modes."""

from wyoming_microsoft_tts.sentence_boundary import SentenceBoundaryDetector


def _stream(sbd: SentenceBoundaryDetector, *chunks: str) -> list[str]:
    """Feed chunks through the detector and return yielded sentences."""
    out: list[str] = []
    for chunk in chunks:
        out.extend(sbd.add_chunk(chunk))
    return out


# Plain-text mode regression — current behaviour must not change.


def test_plain_yields_on_sentence_end_plus_whitespace():
    """Plain mode yields each sentence at end-punct + whitespace."""
    sbd = SentenceBoundaryDetector()
    yielded = _stream(sbd, "Hello. World. ")
    tail = sbd.finish()
    assert yielded == ["Hello.", "World."]
    assert tail == ""


def test_plain_finish_flushes_unterminated_tail():
    """Plain mode: finish() returns buffered text without a sentence boundary."""
    sbd = SentenceBoundaryDetector()
    yielded = _stream(sbd, "Hello world")
    tail = sbd.finish()
    assert yielded == []
    assert tail == "Hello world"


def test_plain_strips_word_asterisks():
    """Plain mode strips *asterisks* around words (markdown emphasis)."""
    sbd = SentenceBoundaryDetector()
    yielded = _stream(sbd, "I am *very* happy. ")
    assert yielded == ["I am very happy."]


# SSML mode: plain-text sentences still split normally.


def test_ssml_plain_text_still_splits():
    """SSML mode falls back to plain-text splitting when no tags are present."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "Hello. World. ")
    tail = sbd.finish()
    assert yielded == ["Hello.", "World."]
    assert tail == ""


def test_ssml_asterisks_are_preserved():
    """SSML mode must not mangle input — asterisks survive untouched."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "*emphatic* word. ")
    assert yielded == ["*emphatic* word."]


# SSML mode: tag-aware splitting.


def test_ssml_per_sentence_wrapping_streams_each_block():
    """Per-sentence prosody blocks stream one yield per block."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(
        sbd, "<prosody rate='slow'>Hi.</prosody> <prosody>Bye.</prosody> "
    )
    assert yielded == [
        "<prosody rate='slow'>Hi.</prosody>",
        "<prosody>Bye.</prosody>",
    ]


def test_ssml_long_span_buffers_until_close():
    """A tag spanning multiple sentences buffers until the close."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "<prosody>First. Second. Third.</prosody> ")
    assert yielded == ["<prosody>First. Second. Third.</prosody>"]


def test_ssml_mixed_plain_and_tagged_text():
    """Plain sentences flanking a tag block each yield separately."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(
        sbd, "Hello. <prosody>World.</prosody> Done. "
    )
    assert yielded == [
        "Hello.",
        "<prosody>World.</prosody>",
        "Done.",
    ]


def test_ssml_self_closing_tag_does_not_trigger_yield():
    """Self-closing tags must not be mistaken for a top-level close."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "Hello <break time='500ms'/> world. ")
    assert yielded == ["Hello <break time='500ms'/> world."]


def test_ssml_self_closing_inside_long_span_does_not_leak():
    """Self-close inside a wrapping tag doesn't bring depth to 0 prematurely."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(
        sbd, "<prosody>Hi.<break time='250ms'/>Bye.</prosody> "
    )
    assert yielded == ["<prosody>Hi.<break time='250ms'/>Bye.</prosody>"]


def test_ssml_nested_tags_buffer_correctly():
    """Yield only when depth returns all the way to 0 across nested tags."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(
        sbd,
        "<mstts:express-as style='cheerful'><prosody rate='fast'>"
        "One. Two.</prosody></mstts:express-as> ",
    )
    assert yielded == [
        "<mstts:express-as style='cheerful'><prosody rate='fast'>"
        "One. Two.</prosody></mstts:express-as>"
    ]


def test_ssml_chunked_across_tag_boundary():
    """Tag-close split between chunks is detected after re-feed."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(
        sbd, "<prosody>Hi. By", "e.</prosody> Next. "
    )
    assert yielded == ["<prosody>Hi. Bye.</prosody>", "Next."]


def test_ssml_chunked_in_middle_of_open_tag():
    """Open tag split between chunks is reassembled before classification."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "Hello. <pros", "ody>World.</prosody> ")
    assert yielded == ["Hello.", "<prosody>World.</prosody>"]


def test_ssml_malformed_close_before_open_clamps_depth():
    """Orphan close tag must not push depth negative; output still passes through."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "</prosody>foo. ")
    assert yielded == ["</prosody>foo."]


def test_ssml_finish_flushes_unterminated_buffer():
    """finish() returns whatever's still buffered, regardless of depth."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "<prosody>Hi.</prosody>")
    tail = sbd.finish()
    assert yielded == []
    assert tail == "<prosody>Hi.</prosody>"


def test_ssml_finish_resets_internal_state():
    """finish() resets depth/tag state so the detector is reusable."""
    sbd = SentenceBoundaryDetector(ssml=True)
    _stream(sbd, "<prosody>Hi.")
    sbd.finish()
    yielded = _stream(sbd, "Hello. ")
    assert yielded == ["Hello."]


def test_ssml_multi_punctuation_yields_once():
    """Chained sentence-end punctuation yields a single sentence."""
    sbd = SentenceBoundaryDetector(ssml=True)
    yielded = _stream(sbd, "Wait?! Really. ")
    assert yielded == ["Wait?!", "Really."]
