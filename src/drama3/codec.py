"""Strict Fish codec loading and fixed-window graph execution."""

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


class CachedMask:
    def __init__(self, make_mask):
        self.make_mask = make_mask
        self.cache = {}

    def __call__(self, length, x_lens=None):
        if length not in self.cache:
            self.cache[length] = self.make_mask(length, x_lens).to("cuda")
        return self.cache[length]


def load_codec(checkpoint: Path, fish_source: Path):
    codec = instantiate(OmegaConf.load(fish_source / "fish_speech/configs/modded_dac_vq.yaml"))
    weights = torch.load(checkpoint / "codec.pth", map_location="cpu", weights_only=True, mmap=True)
    weights = weights.get("state_dict", weights)
    if any(name.startswith("generator.") for name in weights):
        weights = {
            name.removeprefix("generator."): value
            for name, value in weights.items()
            if name.startswith("generator.")
        }
    # Only deterministic buffers that became non-persistent upstream are omitted.
    legacy_buffers = {
        f"{prefix}.{name}"
        for prefix in ("encoder.block.4.block.5", "quantizer.pre_module", "quantizer.post_module")
        for name in ("freqs_cis", "causal_mask")
    }
    codec.load_state_dict(
        {name: value for name, value in weights.items() if name not in legacy_buffers},
        strict=True,
        assign=True,
    )
    codec.eval().to("cuda")
    # Native causal masks depend only on length. Cache the exact tensors before
    # capture instead of recreating CPU tensors and copying them on every call.
    for module in codec.modules():
        if hasattr(module, "make_window_limited_mask") and module.causal:
            module.make_window_limited_mask = CachedMask(module.make_window_limited_mask)
            module.make_mask = CachedMask(module.make_mask)
    return codec


class CodecGraph:
    def __init__(self, codec, batch: int, length: int):
        self.batch = batch
        self.length = length
        self.codes = torch.zeros((batch, 10, length), dtype=torch.int64, device="cuda")
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                self.output = codec.from_indices(self.codes)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph), torch.autocast("cuda", dtype=torch.bfloat16):
            self.output = codec.from_indices(self.codes)

    def run(self, codes: torch.Tensor):
        if codes.shape[0] > self.batch or codes.shape[-1] != self.length:
            raise ValueError("Audio codes must match the captured window length")
        self.codes[: codes.shape[0], :, : codes.shape[-1]].copy_(codes)
        self.graph.replay()
        return self.output[: codes.shape[0], :, : codes.shape[-1] * 2048]
