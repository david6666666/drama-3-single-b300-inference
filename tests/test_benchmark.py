import asyncio

import aiohttp
import pytest
from aiohttp import web

from drama3.benchmark import request, summarize


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
