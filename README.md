# Real-time translated meeting captions

Live captions for a multi-speaker meeting, translated into several languages at once, with speaker labels and a WebVTT export when the meeting ends.

Companion repo for the AssemblyAI blog post "How to build real-time translated captions for meetings".

## How it works

- One streaming session per meeting on **AssemblyAI STT 3.7 Realtime** (`assemblyai-stt-3-7-realtime`) with `speaker_labels: true`.
- `llm_gateway` translates every final turn on the same WebSocket. One prompt asks `qwen3.5-4b-32k-fast` for a JSON object keyed by language, so one gateway call per turn covers every caption language.
- `server.py` joins each `LLMGatewayResponse` to its `Turn` by `turn_order`, keeps the speaker label, and fans captions out to viewers over a WebSocket. Each viewer picks a language.
- If no translation arrives within 3 seconds (the gateway sends nothing when a call fails), viewers see the source text.
- When the audio ends, `captions_out/meeting.<lang>.vtt` is written for every language.

## Files

| File | What it does |
|---|---|
| `server.py` | Streaming session, event handlers, FastAPI fan-out, WebVTT export |
| `translate_prompt.py` | The multi-language prompt and a tolerant JSON parser with fallback |
| `captions.py` | Caption store keyed by `turn_order`, WebVTT writer |
| `audio_source.py` | 16 kHz PCM from a WAV file, any ffmpeg input, or a live capture device |
| `overlay.html` | Minimal browser caption overlay with a language picker |
| `tests/test_offline.py` | Offline tests (no API key needed) |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ASSEMBLYAI_API_KEY=your_api_key_here

# Replay a recorded meeting at real-time pace (16 kHz mono WAV needs no ffmpeg)
python server.py --audio standup.wav --langs es,fr,de

# Any other file or URL goes through ffmpeg
python server.py --audio standup.mp4 --langs es,ja

# Live meeting audio from a virtual audio device (macOS example)
python server.py --audio "ffmpeg:-f avfoundation -i :1" --langs es,fr
```

Open `http://127.0.0.1:8000/?lang=fr` in as many browser tabs as you like. Translated WebVTT is also live at `/captions/fr.vtt`.

The server binds to 127.0.0.1 and has no authentication. Put it behind your own auth before you expose it.

## Tests

```bash
python tests/test_offline.py
```

The tests build `Turn` and `LLMGatewayResponse` messages from the documented JSON shapes, check the connection parameters the SDK sends, the JSON parsing fallbacks, out-of-order translations, the WebVTT output and the per-language WebSocket fan-out. They don't call the API.

## Notes

- Billing: streaming is billed on session duration, and the gateway bills the model's input and output tokens. See [pricing](https://www.assemblyai.com/pricing) and the per-model rates in the [LLM Gateway model list](https://www.assemblyai.com/docs/llm-gateway/available-models).
- The multi-language JSON prompt is this repo's own pattern, not a documented recipe. Spot-check its output on your meetings, and raise `max_tokens` if you add languages.
- Docs: [Real-time translation](https://www.assemblyai.com/docs/streaming/real-time-translation), [Streaming diarization](https://www.assemblyai.com/docs/streaming/label-speakers-and-separate-channels).
