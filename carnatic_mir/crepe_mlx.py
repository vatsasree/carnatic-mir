"""CREPE forward pass in MLX for Apple GPUs. PyTorch's MPS convolutions use under a tenth of the GPU's
throughput on CREPE's long kernels. Weights come from torchcrepe; the output matches it to ~1e-3."""
import os

import numpy as np

_EPS = 0.0010000000474974513
_nets: dict = {}


def _load(model: str, dtype):
    import mlx.core as mx
    import torch
    import torchcrepe

    key = (model, dtype)
    if key not in _nets:
        path = os.path.join(os.path.dirname(torchcrepe.__file__), "assets", f"{model}.pth")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        layers = []
        for i in range(1, 7):
            w = sd[f"conv{i}.weight"].numpy()[..., 0].transpose(0, 2, 1)          # (out, k, in)
            bn = f"conv{i}_BN"
            scale = (sd[f"{bn}.weight"] / torch.sqrt(sd[f"{bn}.running_var"] + _EPS)).numpy()
            shift = sd[f"{bn}.bias"].numpy() - sd[f"{bn}.running_mean"].numpy() * scale
            layers.append(tuple(mx.array(a).astype(dtype) for a in (w, sd[f"conv{i}.bias"].numpy(), scale, shift)))
        head = (mx.array(sd["classifier.weight"].numpy().T).astype(dtype),
                mx.array(sd["classifier.bias"].numpy()).astype(dtype))
        _nets[key] = (layers, head)
    return _nets[key]


def _forward(x, layers, head):
    import mlx.core as mx

    x = x[..., None]                                                               # (B, 1024, 1)
    for i, (w, b, scale, shift) in enumerate(layers):
        pad, stride = ((254, 254), 4) if i == 0 else ((31, 32), 1)
        x = mx.pad(x, [(0, 0), pad, (0, 0)])
        x = mx.maximum(mx.conv1d(x, w, stride=stride) + b, 0) * scale + shift
        x = mx.maximum(x[:, 0::2], x[:, 1::2])                                     # max-pool 2
    x = x.reshape(x.shape[0], -1)
    return mx.sigmoid(x @ head[0] + head[1])


def salience(frames: np.ndarray, model: str = "full", batch: int = 256, dtype: str = "float16") -> np.ndarray:
    """frames: (n, 1024) already mean/std normalised -> (n, 360) float32 saliences."""
    import mlx.core as mx

    dt = getattr(mx, dtype)
    layers, head = _load(model, dt)
    fn = mx.compile(lambda x: _forward(x, layers, head).astype(mx.float32))
    out = np.empty((len(frames), 360), np.float32)
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        n = len(chunk)
        if n < batch:                                                              # keep one compiled shape
            chunk = np.pad(chunk, ((0, batch - n), (0, 0)))
        y = fn(mx.array(chunk).astype(dt))
        mx.eval(y)
        out[i:i + n] = np.array(y)[:n]
    return out
