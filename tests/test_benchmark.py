import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web

from drama3.benchmark import request, request_body, summarize


def test_preencoded_reference_preserves_codes_for_all_backends():
    sample = {
        "text": "Hello",
        "reference_text": "Reference",
        "reference_audio": "/not/read/when/codes/are/present.wav",
        "reference_codes": [[2 * row, 2 * row + 1] for row in range(10)],
    }
    first = request_body(sample, "vllm", 500)
    second = request_body(sample, "standalone", 500)
    third = request_body(sample, "sglang", 500)
    assert first == second
    assert first["ref_audio_codes"] == sample["reference_codes"]
    assert "ref_audio" not in first and "references" not in first
    assert third["references"] == [
        {"vq_codes": sample["reference_codes"], "text": sample["reference_text"]}
    ]
    assert "ref_audio" not in third and "ref_audio_codes" not in third and "ref_text" not in third
    first_generation = {
        key: value for key, value in first.items() if key not in {"ref_audio_codes", "ref_text"}
    }
    assert first_generation == {key: value for key, value in third.items() if key != "references"}


@pytest.mark.parametrize("backend", ["vllm", "standalone", "sglang"])
@pytest.mark.parametrize("codes_field", [{}, {"reference_codes": None}])
def test_raw_reference_remains_audio_only(tmp_path, backend, codes_field):
    audio_bytes = b"reference waveform bytes"
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(audio_bytes)
    sample = {
        "text": "Hello",
        "reference_text": "Reference",
        "reference_audio": str(audio_path),
        **codes_field,
    }
    body = request_body(sample, backend, 500)
    audio = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")
    assert "ref_audio_codes" not in body
    if backend == "vllm":
        assert body["ref_audio"] == audio
        assert body["ref_text"] == sample["reference_text"]
        assert "references" not in body
    else:
        assert body["references"] == [{"audio": audio, "text": sample["reference_text"]}]
        assert "ref_audio" not in body and "ref_text" not in body


def test_frame_throughput_and_rtf_use_different_denominators():
    rows = [
        {"success": True, "audio_samples": 20480, "ttfa_s": 0.1, "rtf": 0.2},
        {"success": True, "audio_samples": 40960, "ttfa_s": 0.3, "rtf": 0.4},
    ]
    result = summarize(rows, 2.0)
    assert result["throughput_frames_s"] == 15
    assert result["ttfa_p50_ms"] == pytest.approx(200)
    assert result["ttfa_p95_ms"] == pytest.approx(290)
    assert result["rtf_mean"] == pytest.approx(0.3)


def test_failure_disqualifies_run_and_remains_in_makespan():
    rows = [
        {"success": True, "audio_samples": 20480, "ttfa_s": 0.1, "rtf": 0.2},
        {"success": False, "error": "timeout"},
    ]
    result = summarize(rows, 10.0)
    assert result["throughput_frames_s"] == 1
    assert result["failed"] == 1
    assert not result["valid"]


@pytest.mark.parametrize(
    "payload,status,content_type,success",
    [
        (b"\x00\x01" * 2048, 200, "audio/pcm", True),
        (b"", 200, "audio/pcm", False),
        (b"\x00", 200, "audio/pcm", False),
        (b'{"error":"failed"}', 200, "application/json", False),
        (b"unavailable", 503, "text/plain", False),
    ],
)
def test_client_validates_actual_response(payload, status, content_type, success):
    async def check():
        async def handler(_):
            return web.Response(body=payload, status=status, content_type=content_type)

        app = web.Application()
        app.router.add_post("/speech", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with aiohttp.ClientSession() as session:
                row, audio = await request(session, f"http://127.0.0.1:{port}/speech", {}, "r", "s")
            assert row["success"] is success
            if success:
                assert row["acoustic_frames"] == 1
                assert audio == payload
            else:
                assert "error" in row
        finally:
            await runner.cleanup()

    asyncio.run(check())
