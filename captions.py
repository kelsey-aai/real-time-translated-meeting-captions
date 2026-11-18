"""Caption store keyed by turn_order, plus WebVTT export per language."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from translate_prompt import parse_translations


@dataclass
class Caption:
    turn_order: int
    speaker: str
    source: str
    start_ms: int
    end_ms: int
    translations: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # pending -> translated | fallback


class CaptionStore:
    def __init__(self, targets: List[str]):
        self.targets = targets
        self._captions: Dict[int, Caption] = {}
        self._lock = threading.Lock()

    def add_final_turn(self, turn_order: int, transcript: str, speaker: Optional[str],
                       words: list) -> Caption:
        start = words[0].start if words else 0
        end = words[-1].end if words else start
        with self._lock:
            cap = self._captions.get(turn_order)
            if cap is None:
                cap = Caption(turn_order, speaker or "PENDING", transcript, start, end)
                self._captions[turn_order] = cap
            else:  # a later final for the same turn replaces the text
                cap.source, cap.speaker, cap.start_ms, cap.end_ms = (
                    transcript, speaker or cap.speaker, start, end)
            return cap

    def add_translation(self, turn_order: int, content: str) -> Optional[Caption]:
        with self._lock:
            cap = self._captions.get(turn_order)
            if cap is None:
                return None
            cap.translations, ok = parse_translations(content, self.targets, cap.source)
            cap.status = "translated" if ok else "fallback"
            return cap

    def fall_back(self, turn_order: int) -> Optional[Caption]:
        """No LLMGatewayResponse arrived in time: show the source text."""
        with self._lock:
            cap = self._captions.get(turn_order)
            if cap is None or cap.status != "pending":
                return None
            cap.translations = {lang: cap.source for lang in self.targets}
            cap.status = "fallback"
            return cap

    def ordered(self) -> List[Caption]:
        with self._lock:
            return [self._captions[k] for k in sorted(self._captions)]

    def to_webvtt(self, lang: str) -> str:
        lines = ["WEBVTT", ""]
        for i, cap in enumerate(self.ordered(), start=1):
            text = cap.translations.get(lang, cap.source)
            end = max(cap.end_ms, cap.start_ms + 500)
            lines += [str(i), f"{_ts(cap.start_ms)} --> {_ts(end)}",
                      f"<v Speaker {cap.speaker}>{text}", ""]
        return "\n".join(lines)


def _ts(ms: int) -> str:
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
