"""Vocal source separation via `audio-separator` (BS-RoFormer / Mel-RoFormer / Demucs ...).

Runs on Apple-Silicon GPU (MPS) automatically when available.

Carnatic caveat: these models are trained on Western pop/rock. The accompanying violin
shadows the vocal line in the same register and can leak into the vocal stem, and the
tanpura drone may leak too. Listen to the vocal stem in the viewer and compare presets.
"""
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MODELS, SEPARATION_PRESETS, SR_MIX

_cache: dict = {}   # model filename -> loaded Separator (keeps model in memory across tracks)


def settings(batch: int | None = None) -> dict:
    """Inference settings. Defaults favour speed on CUDA; override with env vars:
    CARNATIC_MIR_SEP_OVERLAP (RoFormer chunk overlap; model default is 4, we use 2 = half the passes),
    CARNATIC_MIR_SEP_FP16 (1/0, CUDA autocast), CARNATIC_MIR_MLX (1/0, MLX fast path on Apple GPUs).
    CARNATIC_MIR_SEP_BATCH only reaches non-RoFormer MDXC models: audio-separator ignores it for RoFormer."""
    import torch
    cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu"),
        "overlap": int(os.environ.get("CARNATIC_MIR_SEP_OVERLAP", 2)),
        "batch": batch or int(os.environ.get("CARNATIC_MIR_SEP_BATCH", 4 if cuda else 1)),
        "fp16": os.environ.get("CARNATIC_MIR_SEP_FP16", "1" if cuda else "0") == "1" and cuda,
        "mlx": os.environ.get("CARNATIC_MIR_MLX", "1") == "1",
    }


def _maybe_accelerate(sep, cfg: dict) -> None:
    """On Apple GPUs, run BS-RoFormer's transformer layers in MLX (same output, ~1.5x faster)."""
    from . import roformer_mlx
    model = getattr(sep.model_instance, "model_run", None)
    if (cfg["device"] == "mps" and cfg["mlx"] and type(model).__name__ == "BSRoformer"
            and roformer_mlx.available()):
        roformer_mlx.accelerate(model)


def _get_separator(model_file: str, work_dir: Path, cfg: dict):
    from audio_separator.separator import Separator

    key = (model_file, cfg["overlap"], cfg["batch"], cfg["fp16"], cfg["mlx"])
    sep = _cache.get(key)
    if sep is None:
        (MODELS / "separation").mkdir(parents=True, exist_ok=True)
        _cache.clear()                               # keep only one model resident
        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(MODELS / "separation"),
            output_dir=str(work_dir),
            output_format="WAV",
            sample_rate=SR_MIX,
            normalization_threshold=1.0,   # don't rescale stems; keep levels comparable to the mix
            use_autocast=cfg["fp16"],
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": cfg["batch"], "overlap": cfg["overlap"], "pitch_shift": 0},
            demucs_params={"segment_size": "Default", "shifts": 1, "overlap": 0.25, "segments_enabled": True},
        )
        sep.load_model(model_filename=model_file)   # downloads weights on first use
        _maybe_accelerate(sep, cfg)
        _cache[key] = sep
    sep.output_dir = str(work_dir)
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = str(work_dir)
    return sep


def separate_vocals(mix_wav: Path, out_dir: Path, model: str = "bs_roformer", log=print) -> dict:
    """Write `vocals.wav` and `accompaniment.wav` (sum of all non-vocal stems) into out_dir."""
    model_file = SEPARATION_PRESETS.get(model, model)
    work = out_dir / "_sep_tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    cfg = settings()
    log(f"separation: {model_file} on {cfg['device']} (overlap {cfg['overlap']}, batch {cfg['batch']}, "
        f"fp16 {cfg['fp16']}, mlx {cfg['mlx'] and cfg['device'] == 'mps'})")
    try:
        sep = _get_separator(model_file, work, cfg)
        files = sep.separate(str(mix_wav))
    except RuntimeError as e:                      # GPU out of memory -> retry one chunk at a time
        if "out of memory" not in str(e).lower() or cfg["batch"] == 1:
            raise
        import torch
        _cache.clear()
        torch.cuda.empty_cache()
        cfg = settings(batch=1)
        log("separation: out of GPU memory, retrying with batch 1")
        sep = _get_separator(model_file, work, cfg)
        files = sep.separate(str(mix_wav))
    files = [Path(f) if Path(f).is_absolute() else work / f for f in files]

    vocals, others = None, []
    for f in files:
        y, sr = sf.read(str(f), dtype="float32", always_2d=True)
        stem = f.stem.lower()
        if "(vocals)" in stem:
            vocals = y
        else:
            others.append(y)
    if vocals is None:
        raise RuntimeError(f"no vocal stem in separator outputs: {[f.name for f in files]}")

    if others:
        n = min(len(vocals), *(len(o) for o in others))
        accomp = np.sum([o[:n] for o in others], axis=0)
    else:  # single-stem model: accompaniment = residual
        mix, _ = sf.read(str(mix_wav), dtype="float32", always_2d=True)
        n = min(len(mix), len(vocals))
        accomp = mix[:n] - vocals[:n]

    sf.write(str(out_dir / "vocals.wav"), vocals, sr, subtype="PCM_16")
    sf.write(str(out_dir / "accompaniment.wav"), np.clip(accomp, -1, 1), sr, subtype="PCM_16")
    shutil.rmtree(work, ignore_errors=True)
    return {"model": model_file, **cfg}
