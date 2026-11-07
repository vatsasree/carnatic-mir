from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent.parent
# Override with env vars, e.g. on Colab point these at Google Drive so outputs/weights persist.
OUTPUTS = Path(os.environ.get("CARNATIC_MIR_OUTPUTS", ROOT / "outputs"))
MODELS = Path(os.environ.get("CARNATIC_MIR_MODELS", ROOT / "models"))
UPLOADS = Path(os.environ.get("CARNATIC_MIR_UPLOADS", ROOT / "data" / "raw"))

SR_MIX = 44100          # sample rate for stored stems
SR_ANALYSIS = 16000     # sample rate for pitch / frame features (CREPE native rate)
HOP_S = 0.01            # 10 ms analysis hop -> 100 frames / s

# Vocal pitch search range (covers low male mandra sthayi to high female tara sthayi)
FMIN, FMAX = 60.0, 1100.0

# Separation model presets (any model filename supported by `audio-separator` also works)
SEPARATION_PRESETS = {
    "bs_roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",   # BS-RoFormer (viperx), vocals SDR ~12.9
    "mel_roformer": "vocals_mel_band_roformer.ckpt",               # Mel-Band RoFormer (Kim)
    "htdemucs_ft": "htdemucs_ft.yaml",                              # Demucs v4 fine-tuned
}
DEFAULT_SEPARATION = "bs_roformer"

# The 12 svarasthanas in 12-TET cents above Sa. Carnatic intonation deviates from these;
# the histogram peaks report the actual positions.
SVARASTHANAS = [
    (0, "S"), (100, "R1"), (200, "R2/G1"), (300, "R3/G2"), (400, "G3"), (500, "M1"),
    (600, "M2"), (700, "P"), (800, "D1"), (900, "D2/N1"), (1000, "D3/N2"), (1100, "N3"),
]
