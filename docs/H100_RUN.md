# Running the batch on the H100 via GitHub

Laptop → GitHub (code + `data/raw/` audio) → server runs in the vLLM container as root →
GitHub (features + compressed stems in `results/h100/`) → laptop viewer.

Model weights are not in git: the server downloads them on first run (~610 MB).
No sharding is needed: every tracked file is under GitHub's 100 MB limit.

## 1. Laptop: push (once)
1. On github.com create a **private**, empty repo, e.g. `carnatic-mir` (no README/licence).
   Keep it private: the recordings are commercial audio.
2. Push:
   ```bash
   cd ~/Research/ICASSP-workshop
   git remote add origin https://github.com/<you>/carnatic-mir.git
   git push -u origin main
   ```
   GitHub asks for a password: use a personal access token (Settings → Developer settings → Fine-grained token,
   access to this repo only, Contents: read and write).

## 2. Server: run
```bash
tmux new -s carnatic                      # survives SSH disconnects; re-attach with: tmux attach -t carnatic
mkdir -p ~/carnatic && cd ~/carnatic
docker run --gpus all -it --ipc=host --name carnatic --entrypoint bash -v "$PWD":/workspace -w /workspace \
  vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
```
Inside the container (you are root):
```bash
command -v git || (apt-get update && apt-get install -y git)
git clone https://github.com/<you>/carnatic-mir.git && cd carnatic-mir
bash scripts/server_run.sh                # installs ffmpeg + packages, checks GPU, runs urls_new.txt, packs results
```
Defaults: overlap 4 and full precision (best quality), 4 tracks in parallel. Override with e.g.
`WORKERS=6 bash scripts/server_run.sh`. Progress: `tail -f results/h100/logs/worker*.log` (second tmux window).
Re-running skips finished tracks.

Push the results back:
```bash
git config user.name "<you>"; git config user.email "<you@example.com>"
git add results/h100 && git commit -m "H100 results" && git push
```

If the container exits, restart it with `docker start -ai carnatic`; files live in `~/carnatic` on the host.

## 3. Laptop: look at the results
```bash
cd ~/Research/ICASSP-workshop && git pull
conda activate carnatic-mir
CARNATIC_MIR_OUTPUTS=results/h100 python -m app.server     # viewer on http://127.0.0.1:8765
```
`results/h100/summary.csv` has one row per track. Stems are AAC `.m4a`; the viewer plays them directly.

## If something fails
- **No internet on the server except GitHub:** shard the weights on the laptop and push them:
  `python scripts/shard.py split models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt`, then
  `git add -f models/separation/*.yaml models/separation/*.shard* models/separation/*.sha256` and push.
  `server_run.sh` joins and checksums them.
- **A package fails to install in the container:** send the error from step "2/4".
- **YouTube:** not needed on the server; all recordings in `urls_new.txt` are already in `data/raw/`.
  (`1WqAz1Gef3U` is unavailable on YouTube and is reported as failed.)
