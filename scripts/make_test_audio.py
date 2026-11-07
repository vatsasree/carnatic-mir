"""Synthesise Carnatic-like test clips with KNOWN ground truth.

Writes, for style in {soft, hard}:
    data/raw/synthetic_<style>.wav          mix (drone + voice + percussion)
    data/raw/synthetic_<style>_vocals.wav   the clean voice (use with --vocals)
    data/raw/synthetic_<style>.gt.json      tonic, tempo, list of ornament events
soft = breathy voice (steep spectral tilt, noise, slow attacks) + gentle strokes (slow attack, dark)
hard = pressed bright voice (flat tilt, fast attacks)           + sharp strokes (instant attack, bright)
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
FPS = 100
TONIC = 146.83
BPM = 80
rng = np.random.default_rng(0)

# score: (kind, args...)  kinds: hold(c,dur) glide(c0,c1,dur) osc(centre,extent_pp,rate,dur)
#                               sph(c,depth,fwhm) step(c0,c1) rest(dur)
SCORE = [
    ("rest", 1.0),
    ("hold", 0, 1.0), ("glide", 0, 700, 0.4), ("hold", 700, 0.5), ("osc", 700, 60, 6.0, 1.2),
    ("hold", 700, 0.3), ("glide", 700, 400, 0.25), ("hold", 400, 0.3), ("rest", 0.7),
    ("hold", 400, 0.3), ("osc", 450, 220, 4.5, 1.5), ("hold", 200, 0.4), ("rest", 0.7),
    ("hold", 400, 0.25), ("sph", 400, 150, 0.08), ("hold", 400, 0.25), ("step", 400, 500),
    ("hold", 500, 0.25), ("sph", 500, 100, 0.07), ("hold", 500, 0.25), ("step", 500, 700),
    ("hold", 700, 0.25), ("sph", 700, 200, 0.09), ("hold", 700, 0.4), ("rest", 0.7),
    ("glide", 700, 1200, 0.5), ("hold", 1200, 1.5), ("glide", 1200, 700, 0.6), ("osc", 700, 50, 5.5, 0.8),
    ("rest", 0.7),
    ("hold", -500, 1.0), ("glide", -500, 0, 0.35), ("osc", 0, 180, 3.5, 1.2), ("hold", 0, 0.8), ("rest", 1.0),
]


def render_contour():
    c, gt, t = [], [], 0.0

    def add(arr):
        nonlocal t
        c.extend(arr)
        t += len(arr) / FPS

    for ev in SCORE:
        k = ev[0]
        t0 = t
        if k == "rest":
            add([np.nan] * int(ev[1] * FPS))
        elif k == "hold":
            add([ev[1]] * int(ev[2] * FPS))
            if ev[2] >= 0.6:
                gt.append({"type": "elongation", "kind": "plain", "start": t0, "end": t, "cents": ev[1]})
        elif k == "glide":
            n = int(ev[3] * FPS)
            x = np.linspace(0, 1, n)
            add(list(ev[1] + (ev[2] - ev[1]) * (3 * x ** 2 - 2 * x ** 3)))     # smooth S-curve
            gt.append({"type": "glide", "start": t0, "end": t, "from": ev[1], "to": ev[2]})
        elif k == "step":
            add(list(np.linspace(ev[1], ev[2], 4)))                            # 40 ms: a note change, not a glide
        elif k == "osc":
            n = int(ev[4] * FPS)
            tt = np.arange(n) / FPS
            add(list(ev[1] + ev[2] / 2 * np.sin(2 * np.pi * ev[3] * tt)))
            gt.append({"type": "heavy_kampita" if ev[2] >= 150 else "simple_oscillation", "start": t0, "end": t,
                       "centre": ev[1], "extent": ev[2], "rate": ev[3]})
            if ev[4] >= 0.6:
                gt.append({"type": "elongation", "kind": "ornamented", "start": t0, "end": t, "cents": ev[1]})
        elif k == "sph":
            n = int(3 * ev[3] * FPS)
            tt = (np.arange(n) - n / 2) / FPS
            sig = ev[3] / 2.355
            add(list(ev[1] - ev[2] * np.exp(-0.5 * (tt / sig) ** 2)))
            gt.append({"type": "sphurita", "time": t0 + n / 2 / FPS, "cents": ev[1], "depth": ev[2]})
    return np.array(c), gt


def voice(cents_fps, style):
    n = int(len(cents_fps) / FPS * SR)
    tf = np.arange(len(cents_fps)) / FPS
    ts = np.arange(n) / SR
    cents = np.interp(ts, tf, np.nan_to_num(cents_fps, nan=0.0))
    on = np.interp(ts, tf, np.isfinite(cents_fps).astype(float)) > 0.5
    f0 = TONIC * 2 ** (cents / 1200)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    slope, noise, att = (2.2, 0.10, 0.15) if style == "soft" else (0.8, 0.01, 0.02)
    y = sum(h ** -slope * np.sin(h * ph) for h in range(1, 30) if h * f0.mean() < SR / 2.2)
    k = int(att * SR)
    env = np.convolve(on.astype(float), np.ones(k) / k, mode="full")[:n]       # attack ramp of `att` s
    env = np.minimum(env, on + 0.0) if style == "hard" else env * on
    y = y / np.abs(y).max() + noise * rng.standard_normal(n)
    return 0.25 * y * env


def drone(dur):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for ratio, phase in [(0.75, 0.0), (2.0, 0.25), (2.0, 0.5), (1.0, 0.75)]:
        env = np.zeros_like(t)
        for k in np.arange(phase, dur, 1.0):
            i = int(k * SR)
            tt = np.arange(len(t) - i) / SR
            env[i:] = np.maximum(env[i:], (1 - np.exp(-tt / 0.03)) * np.exp(-tt / 1.2))   # soft pluck
        out += env * sum(np.sin(2 * np.pi * TONIC * ratio * h * t) / h ** 1.2 for h in range(1, 12))
    return 0.04 * out / np.abs(out).max()


def percussion(dur, style):
    y = np.zeros(int(dur * SR))
    L = int(0.15 * SR)
    tt = np.arange(L) / SR
    for k, t0 in enumerate(np.arange(0.5, dur - 0.2, 60 / BPM / 2)):        # strokes on half-beats
        f = 200 if k % 2 == 0 else 85
        if style == "soft":
            attack = 1 - np.exp(-tt / 0.006)
            seg = attack * np.exp(-tt / 0.06) * np.sin(2 * np.pi * f * tt)
            seg *= 0.5
        else:
            seg = np.exp(-tt / 0.04) * np.sin(2 * np.pi * f * tt) + 0.6 * rng.standard_normal(L) * np.exp(-tt / 0.004)
        i = int(t0 * SR)
        y[i:i + L] += seg[:len(y) - i]
    return 0.15 * y / np.abs(y).max() * (0.5 if style == "soft" else 1.0)


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    c, gt = render_contour()
    dur = len(c) / FPS
    for style in ("soft", "hard"):
        v = voice(c, style)
        mix = drone(dur)[:len(v)] + v + percussion(dur, style)[:len(v)]
        g = 0.9 / np.abs(mix).max()
        sf.write(outdir / f"synthetic_{style}.wav", np.stack([mix, mix], 1) * g, SR)
        sf.write(outdir / f"synthetic_{style}_vocals.wav", np.stack([v, v], 1) * g, SR)
        (outdir / f"synthetic_{style}.gt.json").write_text(json.dumps(
            {"tonic_hz": TONIC, "beat_bpm": BPM, "stroke_rate_per_s": 2 * BPM / 60,
             "events": gt, "cents_100fps": [None if not np.isfinite(x) else round(float(x), 2) for x in c]}))
        print(f"wrote synthetic_{style} ({dur:.1f} s)")
