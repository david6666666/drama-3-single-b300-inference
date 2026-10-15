"""Check deferred host submission against the fixed dynamic serial reference.

Reuse the established real-input schedule and all code/state/PCM gates. Test
fixtures add the production hook after each actual decode/first-frame launch;
no model step is added, and codec input tensors/events are prepared at the same
chunk boundaries. Closing the fixture queue flushes the final tail as Engine.run
would on idle/exit. Full-code D2H in the underlying diagnostic changes timing
relative to serving, so this is not a throughput or audio-quality benchmark.
"""

import argparse
import json
import os
from pathlib import Path

import validate_dynamic_codec_overlap as replay


class DeferredEngine(replay.Engine):
    instance = None

    def __new__(cls):
        engine = super().__new__(cls)
        engine.defer_codec_submit = True
        engine.codec_submissions = []
        engine.launch_count = 0
        engine.after_launch = False
        engine.prepared_at = {}
        engine.deferred_checks = []
        cls.instance = engine
        return engine

    def emit_audio_batch(self, requests):
        before = len(self.codec_submissions)
        super().emit_audio_batch(requests)
        for _, _, _, ready in self.codec_submissions[before:]:
            self.prepared_at[id(ready)] = self.launch_count

    def flush_codec_submissions(self):
        if self.codec_submissions:
            assert self.audio_queue is not None
            prepared = [self.prepared_at.pop(id(job[3])) for job in self.codec_submissions]
            if self.after_launch:
                assert all(count < self.launch_count for count in prepared)
            self.deferred_checks.append(
                dict(
                    mode="thread" if self.audio_queue.threaded else "poll",
                    batches=len(prepared),
                    launch=self.launch_count,
                    prepared_at=prepared,
                    flushed_after_model_launch=self.after_launch,
                )
            )
        super().flush_codec_submissions()


class DeferredDecodeGraph(replay.DecodeGraph):
    def run(self, requests, hidden=None):
        output = super().run(requests, hidden)
        engine = DeferredEngine.instance
        engine.launch_count += 1
        engine.after_launch = True
        try:
            engine.flush_codec_submissions()
        finally:
            engine.after_launch = False
        return output


class DeferredAudioQueue(replay.AudioQueue):
    def submit(self, codes, rows, callbacks, *, ready=None):
        assert ready is not None, "Every deferred batch must retain its original producer event"
        return super().submit(codes, rows, callbacks, ready=ready)

    def close(self):
        engine = DeferredEngine.instance
        engine.flush_codec_submissions()
        assert not engine.codec_submissions
        super().close()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture", default="dedicated")
    options, _ = parser.parse_known_args()
    if options.capture != "dedicated":
        raise ValueError("Deferred codec validation requires the isolated codec capture stream")
    os.environ["DRAMA3_DEFER_CODEC_SUBMIT"] = "1"
    os.environ["DRAMA3_ASYNC_CODEC"] = "1"
    os.environ["DRAMA3_CODEC_DEDICATED_CAPTURE_STREAM"] = "1"
    # Replace only the diagnostic module's fixtures; model functions, schedule
    # construction and equality checks are reused without modification.
    replay.Engine = DeferredEngine
    replay.DecodeGraph = DeferredDecodeGraph
    replay.AudioQueue = DeferredAudioQueue
    replay.main()
    if options.output and DeferredEngine.instance is not None:
        engine = DeferredEngine.instance
        result = json.loads(options.output.read_text())
        result["deferred_submission_checks"] = engine.deferred_checks
        for mode in ("poll", "thread"):
            assert any(
                check["mode"] == mode and check["flushed_after_model_launch"]
                for check in engine.deferred_checks
            ), f"No delayed {mode} submissions were exercised"
        options.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    deferred_submission_batches=sum(
                        check["batches"] for check in engine.deferred_checks
                    ),
                    passed=result["passed"],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
