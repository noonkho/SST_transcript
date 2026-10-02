"""Convert Chinese text between Simplified and Traditional characters (OpenCC).

Whisper often writes Cantonese in Simplified characters. OpenCC converts by
phrase, not character by character, so ambiguous characters come out right
(头发 → 頭髮 but 发展 → 發展). Traditional uses Hong Kong character forms (s2hk).
"""

from __future__ import annotations

import functools

SCRIPTS = {
    "traditional_hk": ("s2hk", "Traditional Chinese (Hong Kong)"),
    # not offered in the web UI, still accepted by the API for existing clients
    "traditional_tw": ("s2twp", "Traditional Chinese (Taiwan)"),
    "simplified": ("t2s", "Simplified Chinese"),
}


@functools.lru_cache(maxsize=None)
def _converter(script: str):
    import opencc
    return opencc.OpenCC(SCRIPTS[script][0])


def convert_text(text: str, script: str) -> str:
    return _converter(script).convert(text) if text else text


def convert_result(result: dict, script: str) -> dict:
    """A copy of `result` with every line, word and the full text converted."""
    if script not in SCRIPTS:
        raise ValueError(f"script must be one of {list(SCRIPTS)}")
    segments = [_convert_segment(seg, script) for seg in result.get("segments") or []]
    out = dict(result, segments=segments, chinese_script=script)
    out["text"] = convert_text(result.get("text", ""), script)
    return out


def _convert_segment(seg: dict, script: str) -> dict:
    text = seg.get("text", "")
    converted = convert_text(text, script)
    out = dict(seg, text=converted)
    if seg.get("words"):
        out["words"] = _convert_words(seg["words"], text, converted, script)
    return out


def _convert_words(words: list[dict], text: str, converted: str, script: str) -> list[dict]:
    """Convert words using the whole line as context: when the line kept its
    length (the usual case), each word takes the converted characters at its
    own position; otherwise words are converted one by one."""
    if len(converted) == len(text):
        out, cursor = [], 0
        for w in words:
            pos = text.find(w["word"], cursor)
            if pos < 0:
                break
            out.append(dict(w, word=converted[pos:pos + len(w["word"])]))
            cursor = pos + len(w["word"])
        else:
            return out
    return [dict(w, word=convert_text(w["word"], script)) for w in words]
