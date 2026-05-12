"""Guess the sentence boundaries in text."""

from collections.abc import Iterable

import regex as re

SENTENCE_END = r"[.!?…]|[。！？]|[؟]|[।॥]"
ABBREVIATION_RE = re.compile(r"\b\p{L}{1,3}\.$", re.UNICODE)

# Yield on sentence-end + any whitespace. We used to require the next
# sentence's first character (uppercase) in the lookahead, which meant
# single-sentence streamed replies never yielded mid-stream — the only
# audio left the server after SynthesizeStop, costing seconds of
# perceived latency and tripping downstream TTS-request timeouts when
# the LLM produced text slowly. ABBREVIATION_RE in add_chunk still
# stitches "Mr. Smith." back together for 1–3 letter words; longer
# abbreviations ("Calif.") can leak through and get split, but Azure
# synthesizes the fragments fine and the latency win is large.
SENTENCE_BOUNDARY_RE = re.compile(
    rf"(.*?(?:{SENTENCE_END}+))(?=\s)",
    re.DOTALL,
)
WORD_ASTERISKS = re.compile(r"\*+([^\*]+)\*+")
LINE_ASTERICKS = re.compile(r"(?<=^|\n)\s*\*+")

# Sentence-end characters as a set for fast membership checks in the SSML scanner.
_SENTENCE_END_CHARS = frozenset(".!?…。！？؟।॥")


class SentenceBoundaryDetector:
    """Detect sentence boundaries in text."""

    def __init__(self, ssml: bool = False) -> None:
        """Initialize the sentence boundary detector.

        When ``ssml`` is True, the detector treats input as SSML and only
        yields at sentence boundaries that fall at tag-depth 0 outside any
        ``<...>``. A self-contained top-level tag block (depth 1→0 close)
        followed by whitespace is also a valid yield point.
        """
        self.ssml = ssml
        self.remaining_text = ""
        self.current_sentence = ""
        # SSML scan state — persists across add_chunk calls.
        self._scan_pos = 0
        self._depth = 0
        self._in_tag = False
        self._tag_is_close = False
        self._tag_prev_char = ""
        self._closed_top = False

    def add_chunk(self, chunk: str) -> Iterable[str]:
        """Add a chunk of text and yield complete sentences."""
        if self.ssml:
            yield from self._add_chunk_ssml(chunk)
            return

        self.remaining_text += chunk
        while self.remaining_text:
            match = SENTENCE_BOUNDARY_RE.search(self.remaining_text)
            if not match:
                break

            match_text = match.group(0)

            if not self.current_sentence:
                self.current_sentence = match_text
            elif ABBREVIATION_RE.search(self.current_sentence[-5:]):
                self.current_sentence += match_text
            else:
                yield remove_asterisks(self.current_sentence.strip())
                self.current_sentence = match_text

            if not ABBREVIATION_RE.search(self.current_sentence[-5:]):
                yield remove_asterisks(self.current_sentence.strip())
                self.current_sentence = ""

            self.remaining_text = self.remaining_text[match.end() :]

    def finish(self) -> str:
        """Return the remaining text as a single item."""
        if self.ssml:
            text = self.remaining_text.strip()
            self.remaining_text = ""
            self._scan_pos = 0
            self._depth = 0
            self._in_tag = False
            self._tag_is_close = False
            self._tag_prev_char = ""
            self._closed_top = False
            return text

        text = (self.current_sentence + self.remaining_text).strip()
        self.remaining_text = ""
        self.current_sentence = ""

        return remove_asterisks(text)

    def _add_chunk_ssml(self, chunk: str) -> Iterable[str]:
        """SSML-aware boundary detection.

        Yields whenever, at depth 0 outside any tag, we encounter
        whitespace preceded by either a sentence-end character or the
        ``>`` that just closed a top-level tag block.
        """
        self.remaining_text += chunk

        while True:
            yielded_text = self._scan_once()
            if yielded_text is None:
                return
            if yielded_text:
                yield yielded_text

    def _scan_once(self) -> str | None:  # noqa: C901
        """Advance the SSML scanner until one yield or end-of-buffer.

        Returns the sentence to yield (possibly the empty string for
        whitespace-only segments — caller may skip), or ``None`` when
        no further yield is possible from the current buffer.
        """
        text = self.remaining_text
        n = len(text)
        i = self._scan_pos

        while i < n:
            c = text[i]
            if self._in_tag:
                if c == ">":
                    self._handle_tag_close()
                    i += 1
                    continue
                if self._tag_prev_char == "" and c == "/":
                    self._tag_is_close = True
                self._tag_prev_char = c
                i += 1
                continue

            if c == "<":
                self._in_tag = True
                self._tag_is_close = False
                self._tag_prev_char = ""
                self._closed_top = False
                i += 1
                continue

            if c.isspace():
                if self._depth == 0:
                    k = i - 1
                    while k >= 0 and text[k].isspace():
                        k -= 1
                    if k >= 0:
                        last = text[k]
                        if last in _SENTENCE_END_CHARS or self._closed_top:
                            sentence = text[: k + 1].strip()
                            j = i
                            while j < n and text[j].isspace():
                                j += 1
                            self.remaining_text = text[j:]
                            self._scan_pos = 0
                            self._closed_top = False
                            return sentence
                i += 1
                continue

            self._closed_top = False
            i += 1

        self._scan_pos = i
        return None

    def _handle_tag_close(self) -> None:
        """Update depth/_closed_top when scanner consumes a ``>``."""
        if self._tag_is_close:
            if self._depth > 0:
                self._depth -= 1
                self._closed_top = self._depth == 0
            else:
                self._closed_top = False
        elif self._tag_prev_char == "/":
            pass
        else:
            self._depth += 1
            self._closed_top = False
        self._in_tag = False
        self._tag_is_close = False
        self._tag_prev_char = ""


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**."""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERICKS.sub("", text)
    return text
