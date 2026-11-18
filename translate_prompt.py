"""One llm_gateway prompt that returns every caption language at once."""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese (Simplified)", "hi": "Hindi",
}

# {{turn}} and {{history_3}} are filled in by the server for each final turn.
# Keep them exactly as written.
_TEMPLATE = (
    "You translate live meeting captions. Translate the current turn from whatever "
    "language is spoken into each of these languages: __NAMES__.\n"
    "Reply with one JSON object and nothing else. Use exactly these keys: __KEYS__. "
    "Each value is the translation as a plain string. Keep people's names, product "
    "names and numbers as spoken. If the turn is already in a target language, copy it.\n\n"
    "Earlier in the meeting, for context only: {{history_3}}\n\n"
    "Current turn: {{turn}}"
)


def build_prompt(targets: List[str]) -> str:
    names = ", ".join(f"{LANGUAGE_NAMES.get(c, c)} ({c})" for c in targets)
    keys = ", ".join(json.dumps(c) for c in targets)
    return _TEMPLATE.replace("__NAMES__", names).replace("__KEYS__", keys)


def max_tokens_for(targets: List[str]) -> int:
    # A few hundred tokens covers one turn in one language; scale per language.
    return max(400, 250 * len(targets))


def parse_translations(content: str, targets: List[str], source: str) -> Tuple[Dict[str, str], bool]:
    """Return ({lang: text}, ok). Any language the model skipped falls back to
    the source transcript so a caption line never goes blank."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    data = {}
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data, dict):
        data = {}
    out: Dict[str, str] = {}
    ok = True
    for lang in targets:
        value = data.get(lang)
        if isinstance(value, str) and value.strip():
            out[lang] = value.strip()
        else:
            out[lang] = source
            ok = False
    return out, ok
