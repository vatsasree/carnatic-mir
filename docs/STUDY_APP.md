# Listening-study app

A browser experiment for the lullaby study. Participants:
1. consent (open to all ages; under-18s need a parent or guardian's consent)
2. profile + current mood
3. volume setting (headphones recommended, self-reported; no tone check)
4. **Phase 1**: rate 10 snippets (GEMS-9, valence, arousal, liking, familiarity, lyrics understood, free words)
5. **Phase 2**: hear the same 10 snippets again in a new order and hold a button / space bar while they feel *calm or soothing*;
   "Start over" during playback or "Listen again and re-mark" afterwards replaces the marks (all listenings are stored)
6. final questions → completion code

Each song has two non-overlapping snippets (versions 1 and 2); each participant hears one version of one song from each of the 10 pairs, versions balanced per song on the server: **Set A** = lullabies of P1–P5 + non-lullabies of P6–P10,
**Set B** = the reverse. Sets are balanced on the server (overall, or within each age group when
`balance_sets_by_age_group` is true in `study/config.json`); snippet order is random per person;
one random Phase 1 trial carries an attention-check row. Works on laptops and phones.

## Files

| Path | In git | What |
|---|---|---|
| `study/pairs.csv` | yes | pair → lullaby / non-lullaby track |
| `study/snippets.csv` | yes | snippet window (start/end s) and random file token per song |
| `study/config.json` | yes | texts, age groups, sets, GEMS items |
| `study/snippets/*.mp3` | **no** | the audio participants hear (copyrighted) |
| `study/data/` | **no** | `responses.sqlite`, `links.json`, `export/*.csv` (personal data) |
| `experiment/` | yes | server (`server.py`), storage, snippet tool, export, web app (`static/`) |

## 1. Snippets
```bash
python -m experiment.snippets suggest   # proposes a 37-43 s phrase-aligned window per song: the one closest to the
                                        # whole song's profile among windows that are at least 70 % sung
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
- `participants.csv`: profile (including self-reported headphones), age group, set, status (`complete` / `started` / `excluded`), completion code
- `ratings.csv`: one row per participant × snippet, with pair, condition, track and snippet window
- `presses.csv`: one row per hold, `start_s` / `end_s` in snippet seconds (add `snippet_start_s` to align with the
  track's `timeseries.csv`)
- `events.csv`: volume step, final questions

There are no practice trials (removed); `phase2_trials.csv` has `attempts` (listenings) and `earlier_holds` (replaced marks); the analysis uses the final listening. `tab_hidden_during_audio > 0` flags trials where the
participant switched away while the music played.

## Timing
Holds are timed on the audio clock (`AudioContext.currentTime`), so they are relative to what was actually played;
`output_latency_s` (typically 10–40 ms; more over Bluetooth) is stored per trial and can be subtracted.

## 5. Analysis
```bash
python -m experiment.export          # responses -> study/data/export/
python -m analysis.report            # -> study/data/results/: summary.md, h1_*/h2_*/h3_*.csv, fig_*.png
```
- Exclusions are applied exactly as pre-registered (`analysis/common.py`); `participants_with_exclusions.csv` lists the reason for each.
- **H1** (`analysis/h1.py`): mixed model `soothing ~ lullaby + (1|participant) + (1|pair) + (1|song)`; the lullaby term's
  p and CI use t with df = pairs - 1, confirmed by an exact permutation test over the 10 pair differences; per-listener
  and per-pair checks; secondary outcomes with Holm; valence equivalence (TOST, +/- 0.5); sensitivity analyses; H4 age interaction.
- **H2** (`analysis/h2.py`): press curves per snippet (latency-corrected); change in share holding 0-3 s after vs 1 s before
  plain / ornamented held notes, glides, simple oscillations, heavy kampita, phrase onsets and percussion entries, against
  random times in the same snippets (2000 permutations); lagged regression on frame features (exploratory).
- **H3** (`analysis/h3.py`): lullaby minus non-lullaby on snippet and whole-song features, exact paired permutation over pairs.
- Test the pipeline without real data: `python -m analysis.simulate --out /tmp/sim && python -m analysis.report --export /tmp/sim --out /tmp/sim/results --quick`.
  On null simulations the H1 test rejects in 8 % of 60 datasets (z-tests: 13 %), H2 in 3.6 % of 84 tests.
