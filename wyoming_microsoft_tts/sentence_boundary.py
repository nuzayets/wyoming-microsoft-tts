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


class SentenceBoundaryDetector:
    """Detect sentence boundaries in text."""

    def __init__(self) -> None:
        """Initialize the sentence boundary detector."""
        self.remaining_text = ""
        self.current_sentence = ""

    def add_chunk(self, chunk: str) -> Iterable[str]:
        """Add a chunk of text and yield complete sentences."""
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
        text = (self.current_sentence + self.remaining_text).strip()
        self.remaining_text = ""
        self.current_sentence = ""

        return remove_asterisks(text)


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**."""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERICKS.sub("", text)
    return text
