"""Real-time translated meeting captions.

One AssemblyAI STT 3.7 Realtime session per meeting, with speaker labels and
llm_gateway translation into several languages at once. A FastAPI server fans
the captions out to any number of viewers, each in the language they pick.

    export ASSEMBLYAI_API_KEY=...
    python server.py --audio standup.mp4 --langs es,fr,de
    open http://127.0.0.1:8000/?lang=fr
"""
from __future__ import annotations

import argparse
import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from assemblyai.streaming.v3 import (
    LLMGatewayResponseEvent,
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    TerminationEvent,
    TurnEvent,
)
from assemblyai.streaming.v3.models import LLMGatewayConfig, LLMGatewayMessage

from audio_source import SAMPLE_RATE, open_source
from captions import Caption, CaptionStore
from translate_prompt import build_prompt, max_tokens_for

SPEECH_MODEL = "assemblyai-stt-3-7-realtime"  # AssemblyAI STT 3.7 Realtime
TRANSLATION_MODEL = "qwen3.5-4b-32k-fast"
HERE = Path(__file__).parent


class Hub:
    """Tracks connected viewers and their caption language."""

    def __init__(self) -> None:
        self.clients: Dict[WebSocket, str] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def publish(self, build: Callable[[str], dict]) -> None:
        """Thread-safe: called from the SDK's read thread."""
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._send_all(build), self.loop)

    async def _send_all(self, build: Callable[[str], dict]) -> None:
        for ws, lang in list(self.clients.items()):
            try:
                await ws.send_json(build(lang))
            except Exception:
                self.clients.pop(ws, None)


class CaptionPipeline:
    """SDK event handlers -> caption store -> viewers."""

    def __init__(self, store: CaptionStore, hub: Hub, fallback_after: float = 3.0):
        self.store, self.hub, self.fallback_after = store, hub, fallback_after

    def on_turn(self, client, event: TurnEvent) -> None:
        if not event.transcript:
            return
        if not event.end_of_turn:
            msg = {"type": "partial", "turn_order": event.turn_order,
                   "speaker": event.speaker_label, "text": event.transcript}
            self.hub.publish(lambda lang: msg)
            return
        cap = self.store.add_final_turn(event.turn_order, event.transcript,
                                        event.speaker_label, event.words or [])
        msg = {"type": "final", "turn_order": cap.turn_order,
               "speaker": cap.speaker, "text": cap.source}
        self.hub.publish(lambda lang: msg)
        # The gateway sends nothing for a turn whose translation fails or times out.
        timer = threading.Timer(self.fallback_after, self._fallback, args=(cap.turn_order,))
        timer.daemon = True
        timer.start()

    def on_llm_response(self, client, event: LLMGatewayResponseEvent) -> None:
        try:
            content = event.data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""
        cap = self.store.add_translation(event.turn_order, content)
        if cap is not None:
            self._publish_translation(cap)

    def _fallback(self, turn_order: int) -> None:
        cap = self.store.fall_back(turn_order)
        if cap is not None:
            self._publish_translation(cap)

    def _publish_translation(self, cap: Caption) -> None:
        def build(lang: str) -> dict:
            return {"type": "translation", "turn_order": cap.turn_order,
                    "speaker": cap.speaker, "status": cap.status,
                    "text": cap.translations.get(lang, cap.source)}
        self.hub.publish(build)


def make_params(targets: List[str], max_speakers: Optional[int] = None) -> RealTimeParameters:
    return RealTimeParameters(
        sample_rate=SAMPLE_RATE,
        speech_model=SPEECH_MODEL,
        speaker_labels=True,
        max_speakers=max_speakers,
        llm_gateway=LLMGatewayConfig(
            model=TRANSLATION_MODEL,
            messages=[LLMGatewayMessage(role="user", content=build_prompt(targets))],
            max_tokens=max_tokens_for(targets),
        ),
    )


def write_vtt(store: CaptionStore, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    for lang in store.targets:
        (out_dir / f"meeting.{lang}.vtt").write_text(store.to_webvtt(lang), encoding="utf-8")
    print(f"Wrote WebVTT files to {out_dir}/")


def run_transcriber(pipeline: CaptionPipeline, params: RealTimeParameters,
                    audio: str, api_key: str, out_dir: Path) -> None:
    client = RealTimeTranscriber(
        RealTimeTranscriberOptions(api_host="streaming.assemblyai.com"), api_key=api_key)
    client.on(RealTimeEvents.Turn, pipeline.on_turn)
    client.on(RealTimeEvents.LLMGatewayResponse, pipeline.on_llm_response)
    client.on(RealTimeEvents.Error, lambda c, err: print(f"Streaming error: {err}"))
    client.on(RealTimeEvents.Termination, lambda c, e: print(
        f"Session ended: {e.audio_duration_seconds}s audio, "
        f"{e.session_duration_seconds}s billed session time"))
    client.connect(params)
    try:
        client.stream(open_source(audio))
    finally:
        client.disconnect(terminate=True)
        write_vtt(pipeline.store, out_dir)


def create_app(pipeline: CaptionPipeline,
               start_audio: Optional[Callable[[], None]] = None) -> FastAPI:
    store, hub = pipeline.store, pipeline.hub

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        if start_audio is not None:
            threading.Thread(target=start_audio, daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def overlay():
        return FileResponse(HERE / "overlay.html")

    @app.get("/languages")
    async def languages():
        return {"languages": store.targets}

    @app.get("/captions/{lang}.vtt")
    async def vtt(lang: str):
        if lang not in store.targets:
            return PlainTextResponse("unknown language", status_code=404)
        return PlainTextResponse(store.to_webvtt(lang), media_type="text/vtt")

    @app.websocket("/ws")
    async def captions_ws(ws: WebSocket, lang: str):
        await ws.accept()
        if lang not in store.targets:
            await ws.close(code=1008)
            return
        for cap in store.ordered()[-5:]:  # catch a late joiner up
            await ws.send_json({"type": "translation", "turn_order": cap.turn_order,
                                "speaker": cap.speaker, "status": cap.status,
                                "text": cap.translations.get(lang, cap.source)})
        hub.clients[ws] = lang
        try:
            while True:
                await ws.receive_text()  # viewers don't send anything; wait for close
        except WebSocketDisconnect:
            hub.clients.pop(ws, None)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True,
                        help="meeting.wav, a file/URL ffmpeg can read, or 'ffmpeg:<input args>'")
    parser.add_argument("--langs", default="es,fr,de", help="comma-separated caption languages")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--out", default="captions_out")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_key = os.environ["ASSEMBLYAI_API_KEY"]
    targets = [code.strip() for code in args.langs.split(",") if code.strip()]
    pipeline = CaptionPipeline(CaptionStore(targets), Hub())
    params = make_params(targets, args.max_speakers)
    app = create_app(pipeline, start_audio=lambda: run_transcriber(
        pipeline, params, args.audio, api_key, Path(args.out)))
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
