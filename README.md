# Carnatic MIR — vocal feature extraction + viewer

```
audio ─ffmpeg─► mix.wav ─BS-RoFormer (MPS)─► vocals.wav ──CREPE f0 (MPS)──► frames.csv ─┐
   │                    (or --vocals <clean track>)  │                                    │
   │                                                 └─timbre/voice quality─► voice_*.csv ─┤  + tonic
   │                └──────────► accompaniment.wav ─HPSS─► percussion.wav ─► strokes.csv ─┤
   └─ essentia tonic (needs tanpura, so on the mix) ───────────────────────────────────────┘
                                                                        ▼
         melodic.py (ornaments, range, intonation) + voice.py + percussion.py summaries
                                   ──► timeseries.csv, features.json, outputs/summary.csv
```

## Setup (once)
```bash
conda create -n carnatic-mir -c conda-forge python=3.11 ffmpeg
conda activate carnatic-mir
pip install -r requirements.txt
```

## Use
```bash
conda activate carnatic-mir
python -m app.server                                   # viewer http://127.0.0.1:8765 ; compare page /compare
python -m carnatic_mir.cli data/raw/*.mp3              # batch; writes outputs/summary.csv (one row per track)
python -m carnatic_mir.cli mix.wav --vocals vocal.wav  # skip separation, e.g. Saraga multitrack vocal mic
python -m carnatic_mir.cli outputs/<track> --tonic 146.8   # fix tonic; recomputes features in ~1 s
python -m carnatic_mir.youtube "<url>" ["<url>" ...]   # YouTube audio -> data/raw/ (original stream, no re-encode)
python -m carnatic_mir.cli "<url>" --start 10:00 --end 15:00   # download + analyse in one go (--playlist for playlists)
python -m carnatic_mir.youtube --file urls.txt          # batch download; format in urls_example.txt
python -m carnatic_mir.cli --urls urls.txt              # batch download + analyse, per-line 'URL [start end]'
python scripts/make_test_audio.py && python scripts/eval_synthetic.py   # ground-truth sanity check
```

## Running on Google Colab (GPU)
1. On the Mac: `./scripts/pack_for_colab.sh` → `dist/carnatic_mir_code.zip` (code only, ~50 KB).
2. In Google Drive create `MyDrive/carnatic_mir/`; upload the zip there and your recordings into `MyDrive/carnatic_mir/audio/`.
3. Open `colab/carnatic_mir_colab.ipynb` in Colab (File → Upload notebook), set Runtime → GPU, edit `JOBS`, run all.
4. Results land in `MyDrive/carnatic_mir/outputs/`; copy track folders into the local `outputs/` and use the viewer,
   or open the viewer inside Colab (last cells). Separation weights are cached in `MyDrive/carnatic_mir/models/`.

Paths can be redirected anywhere with `CARNATIC_MIR_OUTPUTS`, `CARNATIC_MIR_MODELS`, `CARNATIC_MIR_UPLOADS`.

## Big GPU server via GitHub
See [docs/H100_RUN.md](docs/H100_RUN.md): push, run `scripts/server_run.sh` in the container, pull `results/h100/`.

## Speed on Apple Silicon
With `mlx` installed, BS-RoFormer's transformer layers and CREPE run in MLX instead of PyTorch MPS; the outputs match.
Vocal-stem frames below -50 dB skip CREPE. `CARNATIC_MIR_MLX=0` turns this off.
`CARNATIC_MIR_SEP_OVERLAP=1` halves separation time again, with seams every 8 s: pitch stays within ~2 cents (p95) of overlap 4,
but about 5 % of frames change voicing and the voiced share drops. Keep the default 2 for final runs.

## Features
All thresholds live in `carnatic_mir/melodic.py › PARAMS` and are saved into every `features.json`.

| # | Feature | Definition |
|---|---|---|
| 1 | Tempo | Local tempo from percussion onsets (HPSS of accompaniment), median over percussion-active sections; top-3 tempogram candidates (×2/×4 ambiguity is normal); tempo stability (IQR / median) |
| 2 | Elongations | Held svara ≥ 0.6 s: *plain* (stable pitch) or *ornamented* (oscillation centred on one svara); seconds + beats; share of voiced time; which svaras |
| 3 | Glides | Monotonic pitch movement ≥ 150 ¢, 10–90 % time ≥ 80 ms, outside oscillations/sphurita; direction, extent, slope, from→to svara; glide share of note transitions |
| 4 | Sphurita | Steady svara → dip of 60–350 ¢ lasting ≤ 150 ms → back to the same svara; depth, width, svara |
| 5 | Oscillations | ≥ 4 alternating turning points (≥ 30 ¢ swings) spaced 40–350 ms; extent (peak-to-peak), rate, lower↔upper svara; *simple* < 150 ¢ ≤ *heavy kampita* |
| 6 | Octave range | p1–p99 / p5–p95 in cents and octaves, Hz, lowest/highest svara, share of time in mandra/madhya/tara |
| + | Voice timbre & modulation | harmonic tilt (dB/oct), alpha ratio, Hammarberg, centroid ± variability, flatness, H1–H2, CPP, HNR, jitter/shimmer (Praat, on held notes), pitch flutter & tremolo in held notes, MFCC mean |
| + | Vocal softness | phrase attack time & slope, harmonic tilt, alpha ratio, H1–H2, loudness range, vocal-to-accompaniment level |
| + | Intonation | per-svara position vs ET, spread, skew, kurtosis (held notes); pitch-class histogram peaks |
| 7 | Percussion softness | per stroke: attack time, first-30 ms brightness, crest factor, low/high band share, level; stroke rate, stroke dynamic range, percussion-to-vocal level |

## Validation (synthetic clips with known ornaments, `scripts/eval_synthetic.py`)
Soft and hard clips both: glides 5/5, simple oscillations 2/2, heavy kampita 2/2, sphurita 3/3, elongations 8/8,
0 false positives; tempo 80 bpm (true 80); 11/13 softness measures move in the expected direction.

## Caveats
- Separation models are trained on Western music; violin shadowing and tanpura leak into the vocal stem.
  Bleed lowers HNR/CPP and raises jitter/shimmer, so compare voice quality across similar recordings, or use clean tracks.
- CREPE's 64 ms window under-reads fast, wide kampita by ~10–15 % in extent.
- Spectral centroid rises with breath noise; use harmonic tilt as the brightness/softness cue.
- HPSS sends tonal mridangam strokes partly to the harmonic part; stroke counts in very soft playing are unreliable.
- Absolute loudness depends on mastering; all softness measures are level-invariant or relative within a track.
- Svarasthana labels are 12-TET bins; enharmonic pairs (R2/G1, …) need the raga to disambiguate.
