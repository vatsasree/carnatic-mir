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
%cd /content
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
# To match the H100 run exactly (needed if you compare tracks across the two), use OVERLAP '4' and FP16 '0' (~2x slower).
os.environ.setdefault('CARNATIC_MIR_SEP_OVERLAP', '2')
os.environ.setdefault('CARNATIC_MIR_SEP_BATCH', '4')
os.environ.setdefault('CARNATIC_MIR_SEP_FP16', '1')
os.chdir('/content/code'); sys.path.insert(0, '/content/code')
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
# Jobs run in subprocesses that inherit these variables, so changing them only needs this cell re-run.
"""),
    md("""
## Process recordings
One entry per job: `(file in audio/, start, end)`; use `None` for the whole file. Times are `mm:ss` or seconds.
`WORKERS` tracks run at once in separate processes, so one track's CPU stages (voice, percussion, tonic)
overlap with the next track's GPU stages. 2 suits Colab (2 CPU cores, 16 GB T4); use 1 if the GPU runs out of memory.
Every stage is cached in `outputs/<track>/`; re-running skips finished work. Each job logs a `timings:` line.
"""),
    code("""
import shutil

JOBS = [
    ('A_Lullaby_of_Hope_A_heart_touching_Tamil_Lullaby_Song_by_K_S_Chithra [6svsy8gkxlY].webm', None, None),
    ('Annamayya_Video_Song_-_Alara_Chanchalamaina_-_Nagarjuna_Ramya_Krishnan_Kasturi_F [Mny49DS7uNI].webm', None, None),
    ('Carnatic_Classical_Vocal_Uyyala_Luguma_By_Bombay_S_Jayashree [KNCeKnOlcsg].webm', None, None),
    ('Ghumer_Poree_-_Tagore_Lullaby_with_English_Translation_Bombay_Jayshri [ikPsTFFGucc].webm', None, None),
    ('Idhigo_Bhadradri_-_Lyrical_Lord_Ram_Dr_M_Balamuralikrishna_Sri_Bhadrachala_Ramad [cgFj14NQLWE].webm', None, None),
    ('Jagadhodharana_with_Lyrics_by_Bombay_Jayashree_Purandaradas_Keerthanas [VbwfS3Mb8zI].webm', None, None),
    ('Jasoda [omvGb_SW4nk].webm', None, None),
    ('Kannada_Song_-_Baa_Beega_Manamohanaa_-_P_Leela [BraydJLSNqU].webm', None, None),
    ('Krishna_Nee_Begane_Baaro_-_Memorables_Unnikrishnan_Official_Audio_Song [3hppeAOzNbU].webm', None, None),
    ('Laali_Yugave [_C7ysIMkKow].webm', None, None),
    ('MS_SubbuLakshmi-_Ennaga_Manasuku_Rani-_Nilambari-_Adi-_Thyagaraja [QPuMkuHPeFk].webm', None, None),
    ('MS_Subbulakshmi-_Jagadodharana-_Kapi-_Purandharadasa-Adi [zW55f0JZNPI].webm', None, None),
    ('Man_Mandira_-_Shivam_Mahadevan_Katyar_Kaljat_Ghusli_Shankar_Mahadevan_Sachin_Pil [nXbQ-dItErg].webm', None, None),
    ('Paadunnu_Njaaninnu_Kadampuzha [BGw21MVQWXM].webm', None, None),
    ('Palnā_Mein_Lalnā_Bombay_Jayashri_MoonChild_Lullaby [XtLz0oLUatw].webm', None, None),
    ('Phool_Hai_Chand [DO-pFm9n98Q].webm', None, None),
    ('Radha_Sametha_Krishna_Song_Unnikrishnan_Devotional_Radha_Krishna_Padalgal [zziZm5F530w].webm', None, None),
    ('Rara_Chiinanna_Audio_Song_M_S_Subbulakshmi_Radha_Vishwanathan_Carnatic_Classical [O7z9MT9Yr2I].webm', None, None),
    ('Sadiseyaku_ooh_Gali_Sadi_Seyakogali_Song_Raja_Makutam_NTR_Raja_Sulochana_Old_Son [vLRda8SZ8w0].webm', None, None),
    ('Shambho_Mahadeva_-_Shankar_Mahadevan [DGGW0m-LnBY].webm', None, None),
    ('Shwaas_Mee_Full_Song_-_Hariharan_Kedar_Bhagwat_Dr_D_A_Bhagwat [RR1bIAPM3Ro].webm', None, None),
    ('Sri_Ganesa_Saranam_Song_Bombay_Jayashree_Carnatic_Classical_Vinayagar_Bhakti_Pad [DoMXjplFXFU].webm', None, None),
    ('Ye_Divilo_Virisina_Male_Video_Song_Kanne_Vayasu_Movie_Roja_Ramani_Sharath_Babu [dHL7IJm2ZVM].webm', None, None),
    ('హర_వర_సన_Popular_Ayyappa_Song_by_K_J_Yesudas_Ayyappa_Swamy_Songs_ayyappaswamyson [BOjJGALm2kQ].webm', None, None),
]
SEPARATION = 'bs_roformer'      # or 'mel_roformer', 'htdemucs_ft'
WORKERS = 2

os.makedirs('/content/audio', exist_ok=True)
lines = []
for name, start, end in JOBS:
    src, dst = f'{PROJECT}/audio/{name}', f'/content/audio/{name}'
    if not os.path.exists(src):
        print('MISSING on Drive:', name); continue
    if not os.path.exists(dst):
        shutil.copy(src, dst)                 # read input from local SSD
    lines.append(' '.join([dst] + [t for t in (start, end) if t]))
open('/content/jobs.txt', 'w').write('\\n'.join(lines) + '\\n')
print(len(lines), 'jobs')

# progress: open a terminal (or a second cell) and run  tail -f /content/outputs/logs/worker*.log
!cd /content/code && python -m carnatic_mir.cli --list /content/jobs.txt --workers {WORKERS} --separation {SEPARATION}
!grep -h "timings:\\|failed" {LOCAL_OUT}/logs/worker*.log

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
