"""Apple-GPU fast path for BS-RoFormer: runs the transformer stack (97 % of the model's time) in MLX.

PyTorch's MPS backend reaches only ~25 % of the M-series GPU's throughput on this model. The band split,
mask estimators and STFT stay in PyTorch; `accelerate(model)` swaps just `model.layers`.
"""
import numpy as np
import torch
from torch import nn


def available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return torch.backends.mps.is_available()
    except ImportError:
        return False


_KEYS = ("an", "qkv", "g", "gb", "o", "fn", "f1", "f1b", "f2", "f2b")


class _MLXStack(nn.Module):
    """Stands in for the last slot of a (time, freq) block; applies all axial layers in MLX."""

    def __init__(self, layers, dtype: str = "float32", compile: bool = True):
        super().__init__()
        import mlx.core as mx
        self.mx = mx
        self.dtype = getattr(mx, dtype)

        def w(t):
            return mx.array(t.detach().float().cpu().numpy()).astype(self.dtype)

        def w32(t):
            return mx.array(t.detach().float().cpu().numpy())

        def transformer(tr):
            assert isinstance(tr.norm, nn.Identity), "expected norm_output=False"
            out = []
            for attn, ff in tr.layers:
                out.append(dict(
                    an=w32(attn.norm.gamma), qkv=w(attn.to_qkv.weight).T,
                    g=w(attn.to_gates.weight).T, gb=w(attn.to_gates.bias), o=w(attn.to_out[0].weight).T,
                    fn=w32(ff.net[0].gamma), f1=w(ff.net[1].weight).T, f1b=w(ff.net[1].bias),
                    f2=w(ff.net[4].weight).T, f2b=w(ff.net[4].bias)))
            return out

        self.heads = layers[0][0].layers[0][0].heads
        self.blocks = [(transformer(b[0]), transformer(b[1])) for b in layers]
        mx.eval([list(l.values()) for blk in self.blocks for tr in blk for l in tr])
        self._step = mx.compile(self._layer_fn) if compile else self._layer_fn

    def _layer_fn(self, x, an, qkv_w, g, gb, o, fn, f1, f1b, f2, f2b):
        """Residual stream stays float32; matmuls, attention and GELU run in self.dtype."""
        mx, lp, H = self.mx, self.dtype, self.heads
        B, N, _ = x.shape
        n = mx.fast.rms_norm(x, an, 1e-24).astype(lp)
        qkv = (n @ qkv_w).reshape(B, N, 3, H, -1).transpose(2, 0, 3, 1, 4)
        d = qkv.shape[-1]
        q = mx.fast.rope(qkv[0], d, traditional=True, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(qkv[1], d, traditional=True, base=10000.0, scale=1.0, offset=0)
        a = mx.fast.scaled_dot_product_attention(q, k, qkv[2], scale=d ** -0.5)
        gate = mx.sigmoid(n @ g + gb).transpose(0, 2, 1)[..., None]
        x = x + ((a * gate).transpose(0, 2, 1, 3).reshape(B, N, -1) @ o).astype(mx.float32)
        h = mx.fast.rms_norm(x, fn, 1e-24).astype(lp) @ f1 + f1b
        h = 0.5 * h * (1 + mx.erf(h * 0.7071067811865476))
        return x + (h @ f2 + f2b).astype(mx.float32)

    def _layer(self, x, p):
        return self._step(x, *(p[k] for k in _KEYS))

    def forward(self, x):                               # (t, f, d), batch 1 after the model's packing
        mx = self.mx
        dev, dt = x.device, x.dtype
        y = mx.array(x.detach().float().cpu().numpy())
        for time_tr, freq_tr in self.blocks:
            y = y.transpose(1, 0, 2)                    # (f, t, d): attend along time
            for p in time_tr:
                y = self._layer(y, p)
            y = y.transpose(1, 0, 2)                    # (t, f, d): attend along frequency
            for p in freq_tr:
                y = self._layer(y, p)
            mx.eval(y)
        return torch.from_numpy(np.array(y)).to(dev, dt)


def accelerate(model, dtype: str = "float32", compile: bool = True):
    """Replace model.layers in place. Only for chunks of batch 1, which is how audio-separator runs RoFormer."""
    if getattr(model, "_mlx", False):
        return model
    stack = _MLXStack(model.layers, dtype, compile)
    model.layers = nn.ModuleList([nn.ModuleList([nn.Identity(), stack])])
    model._mlx = True
    return model
