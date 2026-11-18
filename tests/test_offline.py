"""Offline tests: no API key, no network. Run with `python -m pytest tests` or
`python tests/test_offline.py`. Server messages are built from the JSON shapes
in the AssemblyAI real-time translation and streaming diarization docs."""
import json
import os
import struct
import sys
import tempfile
import time
import wave
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from assemblyai.streaming.v3 import LLMGatewayResponseEvent, TurnEvent  # noqa: E402
from assemblyai.streaming.v3._base import _build_uri  # noqa: E402

import audio_source  # noqa: E402
from captions import CaptionStore  # noqa: E402
from server import CaptionPipeline, Hub, create_app, make_params  # noqa: E402
from translate_prompt import build_prompt, parse_translations  # noqa: E402

TARGETS = ["es", "fr", "de"]


def turn(order, text, speaker="A", final=True, start=0, end=1200):
    return TurnEvent.model_validate({
        "type": "Turn", "turn_order": order, "transcript": text, "end_of_turn": final,
        "turn_is_formatted": final, "end_of_turn_confidence": 0.9, "speaker_label": speaker,
        "words": [
            {"text": text.split()[0], "start": start, "end": start + 300,
             "confidence": 0.9, "word_is_final": True, "speaker": speaker},
            {"text": text.split()[-1], "start": end - 300, "end": end,
             "confidence": 0.9, "word_is_final": True, "speaker": speaker},
        ],
    })


def gateway(order, text, content):
    return LLMGatewayResponseEvent.model_validate({
        "type": "LLMGatewayResponse", "turn_order": order, "transcript": text,
        "data": {"model": "qwen3.5-4b-32k-fast", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}]},
    })


def test_prompt_keeps_template_variables():
    p = build_prompt(TARGETS)
    assert "{{turn}}" in p and "{{history_3}}" in p
    assert '"es", "fr", "de"' in p and "French (fr)" in p


def test_parse_translations():
    good = '{"es": "Hola a todos.", "fr": "Bonjour à tous.", "de": "Hallo zusammen."}'
    out, ok = parse_translations(good, TARGETS, "Hi everyone.")
    assert ok and out["fr"] == "Bonjour à tous."
    fenced = "```json\n" + good + "\n```"
    assert parse_translations(fenced, TARGETS, "x")[1]
    out, ok = parse_translations('{"es": "Hola"}', TARGETS, "Hi")
    assert not ok and out["es"] == "Hola" and out["de"] == "Hi"
    out, ok = parse_translations("Hola a todos", TARGETS, "Hi")
    assert not ok and set(out.values()) == {"Hi"}


def test_connection_params_serialize():
    params = make_params(TARGETS, max_speakers=6)
    q = parse_qs(urlsplit(_build_uri("streaming.assemblyai.com", params)).query)
    assert q["speech_model"] == ["assemblyai-stt-3-7-realtime"]
    assert q["speaker_labels"] == ["True"] or q["speaker_labels"] == ["true"]
    gw = json.loads(q["llm_gateway"][0])
    assert gw["model"] == "qwen3.5-4b-32k-fast"
    assert "{{turn}}" in gw["messages"][0]["content"]
    assert "format_turns" not in q and "speaker_labels_revision_interval_ms" not in q
    q2 = parse_qs(urlsplit(_build_uri("streaming.assemblyai.com", make_params(TARGETS))).query)
    assert "max_speakers" not in q2


def test_store_out_of_order_and_webvtt():
    store = CaptionStore(TARGETS)
    store.add_final_turn(0, "Good morning, thanks for joining.", "A",
                         turn(0, "Good morning, thanks for joining.").words)
    store.add_final_turn(1, "Let's review the roadmap.", "B",
                         turn(1, "Let's review the roadmap.", "B", start=1500, end=3200).words)
    store.add_translation(1, '{"es": "Revisemos la hoja de ruta.", "fr": "Passons en revue la feuille de route.", "de": "Gehen wir die Roadmap durch."}')
    assert store.fall_back(0).status == "fallback"
    store.add_translation(0, '{"es": "Buenos días, gracias por venir.", "fr": "Bonjour, merci d\'être là.", "de": "Guten Morgen, danke fürs Kommen."}')
    vtt = store.to_webvtt("es")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.200\n<v Speaker A>Buenos días" in vtt
    assert "00:00:01.500 --> 00:00:03.200\n<v Speaker B>Revisemos" in vtt
    assert vtt.index("Buenos") < vtt.index("Revisemos")


def test_wav_source_chunks():
    path = os.path.join(tempfile.mkdtemp(), "tone.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * 16000)  # 1 s
    t0 = time.monotonic()
    chunks = list(audio_source.wav_chunks(path))
    assert len(chunks) == 20 and len(chunks[0]) == audio_source.CHUNK_BYTES
    assert time.monotonic() - t0 > 0.9  # paced to real time
    assert audio_source.ffmpeg_command(["-i", "m.mp4"], True)[4] == "-re"


def test_fan_out_per_language():
    store = CaptionStore(TARGETS)
    pipeline = CaptionPipeline(store, Hub(), fallback_after=0.3)
    with TestClient(create_app(pipeline)) as client:
        with client.websocket_connect("/ws?lang=fr") as fr, \
                client.websocket_connect("/ws?lang=de") as de:
            time.sleep(0.1)
            pipeline.on_turn(None, turn(0, "Good morning", final=False))
            assert fr.receive_json()["type"] == "partial"
            pipeline.on_turn(None, turn(0, "Good morning, everyone."))
            assert fr.receive_json() == {"type": "final", "turn_order": 0,
                                         "speaker": "A", "text": "Good morning, everyone."}
            pipeline.on_llm_response(None, gateway(0, "Good morning, everyone.",
                '{"es": "Buenos días a todos.", "fr": "Bonjour à tous.", "de": "Guten Morgen zusammen."}'))
            got = fr.receive_json()
            assert got["type"] == "translation" and got["text"] == "Bonjour à tous."
            assert got["status"] == "translated"
            de.receive_json(); de.receive_json()
            assert de.receive_json()["text"] == "Guten Morgen zusammen."
            # No LLMGatewayResponse for turn 1: viewers get the source text.
            pipeline.on_turn(None, turn(1, "Any questions?", speaker="B"))
            fr.receive_json()
            late = fr.receive_json()
            assert late["status"] == "fallback" and late["text"] == "Any questions?"
        assert client.get("/captions/fr.vtt").text.startswith("WEBVTT")
        assert client.get("/captions/xx.vtt").status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
