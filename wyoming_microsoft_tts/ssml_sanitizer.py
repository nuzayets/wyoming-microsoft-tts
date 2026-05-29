"""Sanitize model-produced SSML fragments for Azure consumption.

The model sometimes emits SSML with semantically invalid attributes
(``prosody rate="moderate"`` etc.), wraps content in ``<speak>``/``<voice>``
envelopes that would conflict with the operator-controlled voice, or
includes unknown tags. ``sanitize_ssml_fragment`` parses each fragment with
lxml's recover mode, walks the tree to strip the envelope and coerce
invalid attributes, and returns a body fragment safe to embed inside the
operator's own ``<speak><voice>`` wrapper.
"""

import logging
import re
from html import escape

from lxml import etree

_LOGGER = logging.getLogger(__name__)

# SSML default and Microsoft extension namespaces — both are declared on
# the operator's <speak>, so any redundant declarations on inner elements
# are stripped from the serialized fragment for cleanliness.
_SSML_NS = "http://www.w3.org/2001/10/synthesis"
_MSTTS_NS = "https://www.w3.org/2001/mstts"

# Elements we keep verbatim (attributes preserved). Local names only —
# any namespace they happen to be in is irrelevant for the allowlist.
_ALLOWED_TAGS = frozenset({
    "prosody",
    "break",
    "emphasis",
    "say-as",
    "sub",
    "phoneme",
    "lang",
    "p",
    "s",
    "lexicon",
    "audio",
    "mark",
    "w",
    "token",
    # mstts extensions
    "express-as",
    "backgroundaudio",
    "silence",
    "viseme",
    "ttsembedding",
    "embedding",
})

# Stripped (unwrapped — children retained) because the operator owns the envelope.
_ENVELOPE_TAGS = frozenset({"speak", "voice"})

# Azure accepts these enum values for prosody attrs, plus relative/absolute
# numeric forms (+20%, 1.5, -2st, 20Hz). Aliases map common hallucinations
# back to the closest valid enum; unknown strings get the attribute dropped.
_RATE_ENUM = frozenset({"x-slow", "slow", "medium", "fast", "x-fast", "default"})
_RATE_MAP = {
    "moderate": "medium",
    "normal": "medium",
    "average": "medium",
    "regular": "medium",
    "quick": "fast",
    "rapid": "fast",
    "slowly": "slow",
    "quickly": "fast",
}
_PITCH_ENUM = frozenset({"x-low", "low", "medium", "high", "x-high", "default"})
_PITCH_MAP = {
    "normal": "medium",
    "average": "medium",
    "regular": "medium",
}
_VOLUME_ENUM = frozenset(
    {"silent", "x-soft", "soft", "medium", "loud", "x-loud", "default"}
)
_VOLUME_MAP = {
    "normal": "medium",
    "quiet": "soft",
    "average": "medium",
    "regular": "medium",
}

# Matches +20%, -10%, 1.5, 0.5, 20Hz, +1st, -2st, etc. Anything that parses
# as a signed number with optional unit suffix passes through.
_NUMERIC_VALUE_RE = re.compile(r"^[+-]?\d+(\.\d+)?(%|[A-Za-z]+)?$")

# Strip the leading <?xml ... ?> PI before wrapping; lxml rejects PIs at
# arbitrary positions inside a fragment.
_XML_PI_RE = re.compile(r"<\?xml[^?]*\?>", re.IGNORECASE)

# After serialization, drop redundant default/mstts namespace declarations
# (the operator's <speak> already declares both). Other namespaces are left
# alone so unknown extensions still serialize correctly.
_REDUNDANT_XMLNS_RE = re.compile(
    r'\s+xmlns="' + re.escape(_SSML_NS) + r'"'
    r'|\s+xmlns:[A-Za-z_][\w.-]*="' + re.escape(_MSTTS_NS) + r'"'
)


def sanitize_ssml_fragment(text: str) -> str:
    """Return a fragment safe to embed in the operator's SSML envelope.

    - The model's ``<?xml?>``, ``<speak>``, and ``<voice>`` wrappers are
      stripped (children kept).
    - ``prosody`` attributes with invalid enum values are coerced (e.g.
      ``rate="moderate"`` → ``rate="medium"``) or dropped.
    - Unknown tags are unwrapped (children kept) so unfamiliar markup
      doesn't fail the request.
    - If the input cannot be parsed even in recover mode, the text is
      HTML-escaped so it's at least valid character data.
    """
    if not text or not text.strip():
        return ""

    stripped_pi = _XML_PI_RE.sub("", text)
    # Synthetic root lets us parse fragments with multiple top-level
    # siblings or bare text content. Declaring both namespaces here means
    # the operator's namespaces inherit cleanly to inner elements that
    # have no explicit xmlns of their own.
    wrapped = (
        f'<__root__ xmlns="{_SSML_NS}" xmlns:mstts="{_MSTTS_NS}">'
        f"{stripped_pi}"
        "</__root__>"
    )
    parser = etree.XMLParser(recover=True, remove_comments=True, resolve_entities=False)
    try:
        root = etree.fromstring(wrapped.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError:
        _LOGGER.warning("SSML fragment unparseable, falling back to escaped text")
        return escape(text, quote=False)
    if root is None:
        _LOGGER.warning("SSML fragment produced no parse tree, escaping")
        return escape(text, quote=False)

    _clean(root)

    parts: list[str] = []
    if root.text:
        parts.append(root.text)
    for child in root:
        parts.append(etree.tostring(child, encoding="unicode"))

    joined = "".join(parts)
    return _REDUNDANT_XMLNS_RE.sub("", joined)


def _local_name(tag) -> str:
    """Return the local name of an lxml tag, stripping any {namespace} prefix."""
    if not isinstance(tag, str):
        # Could be a Comment, ProcessingInstruction, etc. — caller handles.
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _clean(elem) -> None:
    """Recursively normalize ``elem`` and its descendants in place.

    Post-order so that when we unwrap a parent its (already-cleaned)
    children become correctly-cleaned siblings in the grandparent.
    """
    for child in list(elem):
        _clean(child)

    if elem.getparent() is None:
        return  # synthetic root — leave alone

    if not isinstance(elem.tag, str):
        # PIs and comments inside the body — drop them.
        _unwrap(elem)
        return

    name = _local_name(elem.tag)

    if name in _ENVELOPE_TAGS:
        _unwrap(elem)
        return

    if name not in _ALLOWED_TAGS:
        _LOGGER.debug("Unwrapping unknown SSML tag <%s>", name)
        _unwrap(elem)
        return

    if name == "prosody":
        _fix_prosody(elem)


def _fix_prosody(elem) -> None:
    """Coerce or strip invalid ``prosody`` attributes in place."""
    for attr, enum, alias in (
        ("rate", _RATE_ENUM, _RATE_MAP),
        ("pitch", _PITCH_ENUM, _PITCH_MAP),
        ("volume", _VOLUME_ENUM, _VOLUME_MAP),
    ):
        if attr not in elem.attrib:
            continue
        original = elem.attrib[attr]
        coerced = _coerce_enum_attr(original, enum, alias)
        if coerced is None:
            _LOGGER.debug("Dropping invalid prosody @%s=%r", attr, original)
            del elem.attrib[attr]
        elif coerced != original:
            _LOGGER.debug("Coercing prosody @%s: %r → %r", attr, original, coerced)
            elem.attrib[attr] = coerced


def _coerce_enum_attr(value: str, enum: frozenset, alias: dict) -> str | None:
    """Return an Azure-valid value, or ``None`` if the attribute should be dropped."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    lower = v.lower()
    if lower in enum:
        return lower
    if lower in alias:
        return alias[lower]
    if _NUMERIC_VALUE_RE.match(v):
        return v
    return None


def _unwrap(elem) -> None:
    """Replace ``elem`` in its parent with its text + children + tail.

    Equivalent in effect to ``lxml.html.HtmlElement.drop_tag`` but works
    on plain ``etree`` elements. Text and tail are merged into the
    surrounding nodes so no character data is lost.
    """
    parent = elem.getparent()
    if parent is None:
        return

    idx = parent.index(elem)
    children = list(elem)

    head_text = elem.text or ""
    if idx == 0:
        parent.text = (parent.text or "") + head_text
    else:
        prev = parent[idx - 1]
        prev.tail = (prev.tail or "") + head_text

    for i, child in enumerate(children):
        parent.insert(idx + i, child)

    tail_text = elem.tail or ""
    if children:
        last = children[-1]
        last.tail = (last.tail or "") + tail_text
    elif idx == 0:
        parent.text = (parent.text or "") + tail_text
    else:
        prev = parent[idx - 1]
        prev.tail = (prev.tail or "") + tail_text

    parent.remove(elem)
