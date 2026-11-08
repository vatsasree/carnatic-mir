# Listening-study app

A browser experiment for the lullaby study. Participants:
1. consent
2. profile + current mood
3. volume setting
4. headphone check (Woods et al. 2017)
5. **Phase 1**: rate 10 snippets (GEMS-9, valence, arousal, liking, familiarity, lyrics understood, free words)
6. **Phase 2**: hear the same 10 snippets again in a new order and hold a button / space bar while they feel *calm or soothing*
7. final questions → completion code

Each participant hears one song from each of the 10 pairs: **Set A** = lullabies of P1–P5 + non-lullabies of P6–P10,
**Set B** = the reverse. Sets are balanced within each age group on the server; snippet order is random per person;
one random Phase 1 trial carries an attention-check row. Works on laptops and phones.

## Files

| Path | In git | What |
|---|---|---|
| `study/pairs.csv` | yes | pair → lullaby / non-lullaby track |
| `study/snippets.csv` | yes | snippet window (start/end s) and random file token per song |
| `study/config.json` | yes | texts, age groups, sets, scales, headphone-check rules |
| `study/snippets/*.mp3` | **no** | the audio participants hear (copyrighted) |
| `study/data/` | **no** | `responses.sqlite`, `links.json`, `export/*.csv` (personal data) |
| `experiment/` | yes | server (`server.py`), storage, snippet tool, export, web app (`static/`) |

## 1. Snippets
```bash
python -m experiment.snippets suggest   # proposes a 30-45 s window per song (phrase boundaries, most singing)
```
Check each window in the viewer (`./scripts/start_viewer.sh`, open the track, look at start_s..end_s). Edit
`start_s` / `end_s` in `study/snippets.csv` where needed, then:
```bash
python -m experiment.snippets cut       # -> study/snippets/<token>.mp3, -23 LUFS, 0.5 s fades
```
`suggest` keeps windows you edited; `suggest --force` recomputes all. Re-run `cut` after any edit.

## 2. Before going live
- Replace the two `REPLACE` texts in `study/config.json` (consent text approved by ethics; contact).
- Make one private link per recruitment channel:
  `python -m experiment.links add pilot` (or `age-18-25-whatsapp`, …). The link is `<server>/?k=<key>`.

## 3. Run
```bash
./scripts/start_study.sh            # local: http://127.0.0.1:8770/?k=<key>
./scripts/start_study.sh --public   # pilot over a temporary trycloudflare.com URL
```
For the main study run `python -m experiment.server` behind HTTPS on a server you control
(`HOST=0.0.0.0 PORT=8770`), copying `study/snippets/` and `study/data/links.json` there; back up
`study/data/responses.sqlite` daily.

Live counts: `curl -H "X-Admin-Token: $STUDY_ADMIN_TOKEN" <server>/admin/status` (set `STUDY_ADMIN_TOKEN` when starting the server).

## 4. Export
```bash
python -m experiment.export
```
- `participants.csv`: profile, age group, set, headphone result, status (`complete` / `started` / `excluded`), completion code
- `ratings.csv`: one row per participant × snippet, with pair, condition, track and snippet window
- `presses.csv`: one row per hold, `start_s` / `end_s` in snippet seconds (add `snippet_start_s` to align with the
  track's `timeseries.csv`)
- `events.csv`: volume step, final questions

Practice trials have `condition = practice`; drop them in the analysis. `tab_hidden_during_audio > 0` flags trials where the
participant switched away while the music played.

## Timing
Holds are timed on the audio clock (`AudioContext.currentTime`), so they are relative to what was actually played;
`output_latency_s` (typically 10–40 ms; more over Bluetooth) is stored per trial and can be subtracted.
