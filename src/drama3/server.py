"""Loopback streaming API for the standalone Drama 3 engine."""

import argparse
import asyncio
import queue
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StrictInt, model_validator

from drama3.engine import Engine, Request, decode_reference


class Reference(BaseModel):
    audio: str
    text: str = Field(min_length=1)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    references: list[Reference] = Field(default_factory=list, max_length=1)
    model: str = "fishaudio/drama-3"
    stream: bool = True
    stream_format: str = "audio"
    response_format: str = "pcm"
    max_new_tokens: int = Field(default=500, ge=1, le=500)
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 30
    ref_audio_codes: list[list[StrictInt]] | None = None
    ref_text: str | None = None

    @model_validator(mode="after")
    def validate_reference_codes(self):
        if self.ref_audio_codes is not None:
            codes = self.ref_audio_codes
            if self.references or not self.ref_text or not self.ref_text.strip():
                raise ValueError("Reference codes require ref_text and cannot accompany references")
            if len(codes) != 10 or not 1 <= len(codes[0]) <= 1292:
                raise ValueError("Reference codes must have 10 nonempty rows, up to 60 seconds")
            for index, row in enumerate(codes):
                if len(row) != len(codes[0]):
                    raise ValueError("Reference code rows must have equal lengths")
                if min(row) < 0 or max(row) >= (4096 if index == 0 else 1024):
                    raise ValueError("Reference code is outside its codebook")
        return self


def create_app(checkpoint: str, fish_source: str, slots: int, max_length: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = Engine(checkpoint, fish_source, slots, max_length)
        app.state.engine = engine
        await asyncio.to_thread(engine.ready.wait)
        if engine.failure is not None:
            engine.close()
            raise RuntimeError("Model initialization failed") from engine.failure
        try:
            yield
        finally:
            await asyncio.to_thread(engine.close)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        engine = app.state.engine
        if engine.failure is not None or not engine.ready.is_set():
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {
            "status": "ready",
            "model": "fishaudio/drama-3",
            "slots": slots,
            "dense_precision": engine.model.dense_precision,
            "gdn_state_precision": engine.model.state_precision,
            "moe_layout": engine.model.moe_layout,
            "prefill_moe": engine.model.prefill_moe,
            "prefill_gdn": engine.model.prefill_gdn,
            "prefill_spans": engine.prefill_spans,
            "codec_precision": engine.codec_precision,
            "packed_moe_projection": engine.model.packed_moe_projection,
            "prefill_conv": engine.model.prefill_conv_backend,
            "tuned_prefill_moe": engine.model.tuned_prefill_moe,
            "prefill_max_batch": engine.prefill_max_batch,
            "async_codec": engine.async_codec,
            "audio_delivery_thread": engine.audio_delivery_thread,
        }

    @app.post("/v1/audio/speech")
    async def speech(payload: SpeechRequest):
        if (payload.response_format, payload.stream_format, payload.stream) != (
            "pcm",
            "audio",
            True,
        ):
            raise HTTPException(400, "This endpoint returns streaming PCM audio")
        if (payload.temperature, payload.top_p, payload.top_k) != (1.0, 0.9, 30):
            raise HTTPException(400, "This build implements temperature=1, top_p=0.9, top_k=30")
        try:
            reference = (
                decode_reference(payload.references[0].audio) if payload.references else None
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        request = Request(
            request_id=uuid.uuid4().hex,
            text=payload.input,
            reference=reference,
            reference_text=payload.references[0].text
            if payload.references
            else payload.ref_text or "",
            max_frames=payload.max_new_tokens,
            loop=asyncio.get_running_loop(),
            output=asyncio.Queue(),
            reference_codes=payload.ref_audio_codes,
        )
        try:
            app.state.engine.submit(request)
        except (queue.Full, RuntimeError) as error:
            raise HTTPException(503, str(error)) from error
        try:
            first = await request.output.get()
        except asyncio.CancelledError:
            request.cancelled.set()
            raise
        if isinstance(first, Exception):
            raise HTTPException(500, str(first)) from first
        if first is None:
            raise HTTPException(500, "No audio generated")

        async def chunks():
            try:
                yield first
                while True:
                    chunk = await request.output.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk
            finally:
                request.cancelled.set()

        return StreamingResponse(
            chunks(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "44100",
                "X-Channels": "1",
                "X-Bit-Depth": "16",
                "X-Request-ID": request.request_id,
            },
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fish-source", required=True)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    app = create_app(args.checkpoint, args.fish_source, args.slots, args.max_length)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
