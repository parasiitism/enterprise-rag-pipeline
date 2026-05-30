from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.$:/-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]
