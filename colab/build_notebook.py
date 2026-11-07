"""Generates colab/carnatic_mir_colab.ipynb (edit the cells here, then run this script)."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Carnatic MIR — run the pipeline on a Colab GPU

**Runtime → Change runtime type → GPU** (T4 is fine; L4/A100 faster).

Google Drive layout (create once, `MyDrive/carnatic_mir/`):
```
carnatic_mir/
  carnatic_mir_code.zip   <- made on the Mac by scripts/pack_for_colab.sh
  audio/                  <- put recordings here
  models/                 <- separation weights are cached here after the first run
  outputs/                <- results (one folder per track); sync back to the Mac to use the viewer
```
"""),
    code("""
!nvidia-smi -L
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
import os
PROJECT = '/content/drive/MyDrive/carnatic_mir'
for d in ('audio', 'models', 'outputs'):
    os.makedirs(f'{PROJECT}/{d}', exist_ok=True)
print(sorted(os.listdir(f'{PROJECT}/audio')))
"""),
    md("""
## Get the code
Option A (default): unzip the package uploaded to Drive.
Option B: if the project is on GitHub, comment A and use `git clone`.
"""),
    code("""
!rm -rf /content/code && mkdir -p /content/code
!unzip -oq "$PROJECT/carnatic_mir_code.zip" -d /content/code
# !git clone https://github.com/<you>/<repo>.git /content/code
!ls /content/code
"""),
    code("""
!pip install -q -r /content/code/requirements-colab.txt
# If pip reports it upgraded numpy, use Runtime -> Restart session, then continue from the next cell.
"""),
    code("""
import os, sys, torch
PROJECT = '/content/drive/MyDrive/carnatic_mir'
LOCAL_OUT = '/content/outputs'           # Colab's local SSD: much faster than writing stems to Drive
os.makedirs(LOCAL_OUT, exist_ok=True)
os.environ['CARNATIC_MIR_OUTPUTS'] = LOCAL_OUT
os.environ['CARNATIC_MIR_MODELS'] = f'{PROJECT}/models'      # weights cached on Drive across sessions
# Speed/quality knobs (defaults shown): RoFormer overlap 2 (4 = model default, ~2x slower, slightly cleaner),
# fp16 autocast on CUDA. SEP_BATCH has no effect on RoFormer models (audio-separator runs them one chunk at a time).
os.environ.setdefault('CARNATIC_MIR_SEP_OVERLAP', '2')
os.environ.setdefault('CARNATIC_MIR_SEP_BATCH', '4')
os.environ.setdefault('CARNATIC_MIR_SEP_FP16', '1')
os.chdir('/content/code'); sys.path.insert(0, '/content/code')
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
# Note: these env vars are read when carnatic_mir is first imported; after changing them, restart the session.
"""),
    md("""
## Process recordings
One entry per job: `(file in audio/, start, end)`; use `None` for the whole file. Times are `mm:ss` or seconds.
Jobs run inside this notebook process, so the separation and pitch models are loaded once for all tracks.
Every stage is cached in `outputs/<track>/`; re-running skips finished work. Each job prints a timing line.
"""),
    code("""
import shutil, time, traceback
from carnatic_mir.pipeline import process
from carnatic_mir.cli import _ts, collect_summary

JOBS = [
    ('Moksamu Galada Bhuvilo- Saramati- Adi- Thyagaraja- ML Vasanthakumari - Carnatic Ecstasy.mp3', '10:00', '15:00'),
    ('Omanathinkalkidavo.mp3', None, None),
    ('AUDIO-2026-07-02-12-49-26.mp3', None, None),
]
SEPARATION = 'bs_roformer'      # or 'mel_roformer', 'htdemucs_ft'

os.makedirs('/content/audio', exist_ok=True)
for name, start, end in JOBS:
    src, dst = f'{PROJECT}/audio/{name}', f'/content/audio/{name}'
    if not os.path.exists(src):
        print('MISSING on Drive:', name); continue
    if not os.path.exists(dst):
        shutil.copy(src, dst)                 # read input from local SSD
    print(f'\n=== {name} {start or ""}-{end or ""}')
    try:
        process(dst, separation=SEPARATION, start=_ts(start) if start else None, end=_ts(end) if end else None)
    except Exception:
        traceback.print_exc()
print('summary ->', collect_summary())

shutil.copytree(LOCAL_OUT, f'{PROJECT}/outputs', dirs_exist_ok=True)   # back up to Drive
print('copied results to', f'{PROJECT}/outputs')
"""),
    code("""
import pandas as pd
pd.read_csv(f'{LOCAL_OUT}/summary.csv').set_index('track_id').T
"""),
    md("""
## Look at the results
**On the Mac (recommended):** copy `MyDrive/carnatic_mir/outputs/<track>` folders into the project's `outputs/`
(Google Drive for desktop, or download the zip from the cell below), then run `python -m app.server`.

**Inside Colab:** the next cell starts the viewer and opens it in a new browser tab.
"""),
    code("""
import threading, uvicorn
from app.server import app
threading.Thread(target=lambda: uvicorn.run(app, host='127.0.0.1', port=8765, log_level='warning'), daemon=True).start()
from google.colab import output
output.serve_kernel_port_as_window(8765)
"""),
    code("""
# Optional: zip all outputs for download
!cd "$PROJECT" && zip -qr /content/outputs.zip outputs && ls -lh /content/outputs.zip
from google.colab import files
files.download('/content/outputs.zip')
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
dst = Path(__file__).with_name("carnatic_mir_colab.ipynb")
dst.write_text(json.dumps(nb, indent=1))
print("wrote", dst)
