"""GPU device selection shared by the local embedding and reranking models.

torch.cuda.is_available() only checks that CUDA is compiled in and a device
is visible -- it doesn't catch a broken or mismatched NVIDIA driver, which
instead surfaces as a runtime error the first time a tensor actually moves
to the GPU. Probe with a real op here so a bad driver falls back to CPU up
front, once, instead of crashing mid-query.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def probe_gpu() -> tuple[bool, str | None, str | None]:
    """Return (usable, gpu_name, error).

    `gpu_name` is set whenever a CUDA device is visible, even if the probe
    op below then fails -- callers can use it to report a broken GPU
    instead of just an absent one. `error` holds the driver failure, if any.
    """
    try:
        import torch
    except ImportError:
        return False, None, None

    if not torch.cuda.is_available():
        return False, None, None

    name = torch.cuda.get_device_name(0)
    try:
        (torch.zeros(1, device="cuda") + 1).cpu()
    except Exception as exc:
        return False, name, str(exc)
    return True, name, None


def get_device() -> str:
    """"cuda" if a GPU is present and passes a real op, else "cpu"."""
    usable, _, _ = probe_gpu()
    return "cuda" if usable else "cpu"
