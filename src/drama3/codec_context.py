"""Keep native codec capture inside an explicitly requested CUDA SM context."""

import ctypes
from functools import wraps

import torch


class _CurrentContext:
    def __init__(self):
        driver = ctypes.CDLL("libcuda.so.1")
        self.get = driver.cuCtxGetCurrent
        self.get.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.get.restype = ctypes.c_int
        self.set = driver.cuCtxSetCurrent
        self.set.argtypes = [ctypes.c_void_p]
        self.set.restype = ctypes.c_int

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"CUDA context operation failed with driver status {status}")

    def preserve(self, function):
        @wraps(function)
        def call(*args, **kwargs):
            before = ctypes.c_void_p()
            self.check(self.get(ctypes.byref(before)))
            try:
                return function(*args, **kwargs)
            finally:
                after = ctypes.c_void_p()
                self.check(self.get(ctypes.byref(after)))
                if before.value != after.value:
                    self.check(self.set(before))

        return call


def capture_codec_with_sm_limit(graphs, device, num_sms):
    """Recapture native kernels on a green stream; changing replay streams is insufficient."""
    context_type = getattr(torch.cuda, "GreenContext", None)
    if context_type is None or not context_type.SUPPORTED:
        raise RuntimeError("Codec SM limiting requires CUDA GreenContext support")
    if num_sms <= 0:
        raise ValueError("Codec SM limit must be positive")
    context = context_type.create(num_sms=num_sms, device_id=torch.cuda.current_device())
    stream = context.Stream()
    # Inductor's device guard calls set_device(0), which can replace the green
    # current context with the primary context. Restore it before native cuDNN
    # convolutions continue. No compiled arithmetic or precision is changed.
    compiled = graphs.decoder._transformer_layers_ring
    graphs.decoder._transformer_layers_ring = _CurrentContext().preserve(compiled)
    context.set_context()
    try:
        with torch.cuda.stream(stream):
            graphs.capture(device, capture_stream=stream)
    finally:
        context.pop_context()
    # Both the graph nodes' resource assignment and the capture stream must
    # outlive all replays. The model continues using its full-device context.
    graphs.green_context = context
    graphs.codec_sm_limit = num_sms
