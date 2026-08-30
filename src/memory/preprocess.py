"""One preprocessing path, serving both memory writes and search queries — a query and
the memory it should match have to land in the same embedding space.

Deliberately conservative: normalize, never destroy. Names, dates, and project terms
must survive (a property test holds this), so there is no stemming, no stop-word
removal, no keyword-only normalization. Bump PREPROCESS_VERSION when the rules change;
stored vectors record it, which is what makes a stale vector detectable later.
"""

from __future__ import annotations

import unicodedata

PREPROCESS_VERSION = 1

_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF])


def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t")
    return " ".join(text.split())
