# Lullaby Listening Study: Experiment Design

Nov 8, 2026 · @Vatsa

## Summary

We test whether Indian lullabies make listeners feel more soothed than matched non-lullabies, and which musical moments cause that feeling. About 140 adults in four age groups each hear 10 snippets (one song from each of 10 lullaby / non-lullaby pairs), rate how each made them feel, then hear them again and mark the moments that felt calm or soothing.

**Research questions**

1. Do lullabies evoke more tenderness and peacefulness, and lower arousal, than matched non-lullabies? (H1, primary)
2. Which musical events (held notes, glides, oscillations, percussion, soft phrase onsets) precede moments that listeners mark as soothing? (H2)
3. Do lullabies differ acoustically from non-lullabies on the extracted features? (H3)
4. Does the lullaby effect change with age? (H4, exploratory)

**Design at a glance**

| Element | Plan |
| --- | --- |
| Stimuli | 10 pairs = 20 songs: Carnatic, devotional and film songs in Telugu, Kannada, Tamil, Malayalam, Hindi, Marathi and Gujarati |
| Snippets | \~40 s, cut at phrase boundaries, loudness-matched to -23 LUFS |
| Per participant | 10 snippets: 5 lullabies + 5 non-lullabies, never both songs of a pair |
| Counterbalancing | Set A / Set B, balanced within each age group; random order per person |
| Phase 1 | Rate each snippet: GEMS-9, valence, arousal, liking, familiarity, lyric understanding, free words |
| Phase 2 | Re-listen; hold a button while it feels calm or soothing (time-stamped on the audio clock) |
| Participants | 4 adult age groups x \~30 completers (recruit \~35 each) |
| Setting | Online, headphones required (checked), \~35 minutes |
| Software | Custom web app in the repo (experiment/), built and tested |
| Analysis | Mixed-effects models (H1, H4), event-locked press analysis against MIR features (H2), corpus comparison (H3) |

Items still to decide are collected in the Decisions section with a recommended default for each.

## Revised plan for the workshop deadline (9 Nov 2026)

Data are collected over about 18 hours for a Wednesday deadline, so the study runs as an exploratory version of the full design. These changes override the sections below where they differ; they are fixed before data collection.

| Item | Full design (below) | What we run now |
| --- | --- | --- |
| Participants | 4 age groups x 30 (recruit \~140) | 30-40 completers overall; minimum 20 (10 per set). Age recorded, reported descriptively; H4 not claimed |
| Balancing | Set A / B within each age group | Set A / B balanced overall |
| Snippets per song | 2 recommended | 1 per song |
| Snippet selection | Most representative, at least 70 % sung | Implemented: phrase-aligned 37-43 s window closest to the whole song's profile (held-note share, steady / oscillation / kampita / glide shares, p5-p95 pitch range, pitch movement, percussion activity), among windows at least 70 % sung |
| Snippet length | 40 s +/- 3 s | Lullabies mean 40.1 s, non-lullabies 39.7 s; singing 87 % vs 83 % |
| Rating form | GEMS-9 + valence + arousal + liking | 5 GEMS items (Peacefulness, Tenderness, Sadness, Joyful activation, Tension) + valence + arousal + liking + familiarity + lyrics + optional words |
| Primary outcome | mean(Peacefulness, Tenderness) | Unchanged |
| Phase 2 | Hold while calm or soothing | Unchanged |
| Session length | \~35 min | \~25 min |
| Pre-registration | OSF | This document's timestamped git commit before launch |
| Hosting | Institution server | Laptop + Cloudflare quick tunnel for the collection window (laptop plugged in, sleep disabled) |
| Pilot | 5-8 people in person | 2-3 people, 30 minutes, before launch |

**Why multiple pairs per person, not one full-song pair.** One pair per person gives one noisy comparison per listener: if the lullaby does not soothe that person, song, person and chance cannot be told apart, and with \~30 people each pair would get about 3 listeners. Hearing 5 lullabies and 5 non-lullabies averages over songs within each person, gives each song \~15-20 listeners, and never reveals a pair's contrast. Full songs (2-15 min) would also make song length a confound.

**Why varied timbre and raga across pairs is acceptable.** Within a pair, matching holds singer and style constant. Across pairs, variety shows the effect is not tied to one voice or raga; random effects for pair and song absorb song-to-song differences. The five same-singer pairs (P3, P5, P6, P9, P10) are the main sensitivity check; raga, singer and language are recorded per song.

**Flagged snippet:** P4 non-lullaby (Idhigo Bhadradri) has little singing overall; its best window is 61 % sung. Listen before launch and decide whether to keep it.

**Ethics:** confirm with the institution whether an anonymous online listening study needs approval or an exemption before data are used in the paper.

## Background and rationale

Lullabies are found in almost every culture and share recognisable traits: slow tempo, narrow pitch range, repetition, long held notes and soft, gradual onsets. Listeners across cultures can often identify a lullaby without knowing the language. Most of this work rates whole songs or uses Western stimuli; little of it ties the felt response to specific moments in the music, and almost none uses Indian vocal traditions with their characteristic ornaments (gamakas: glides, kampita oscillations, sphurita).

Our contribution is the combination of two things:

- **A measured description of each recording.** The MIR pipeline (BS-RoFormer vocal separation, CREPE pitch tracking, ornament detection, voice-quality and percussion measures) gives a time-stamped account of what happens in the music: elongations, glides, sphurita, simple oscillations vs heavy kampita, range, tempo, vocal softness and percussion softness.
- **A time-stamped listener response.** Phase 2 records when listeners feel soothed, on the same clock as the features.

Together these let us move from "lullabies feel soothing" to which events in the vocal line are followed by the soothing response.

**Why matched pairs.** A lullaby and a non-lullaby can differ in many ways that have nothing to do with being a lullaby: singer, language, era, recording quality, instrumentation. Each pair holds as many of these constant as possible (same singer where available), so the remaining difference is mainly the lullaby function and style.

**Why felt emotion.** The question is whether lullabies soothe the listener, so we ask how the music made the listener feel, not what emotion the music expresses. These two can differ (sad music can feel pleasant).

## Hypotheses

H1 is the primary, confirmatory hypothesis; H2 and H3 are secondary; H4 is exploratory. All directions are stated before data collection and go into the pre-registration.

| ID | Status | Prediction | Measure | Test |
| --- | --- | --- | --- | --- |
| H1a | Primary | Lullabies are rated higher on tenderness and peacefulness than the matched non-lullaby | Soothing score = mean of GEMS Peacefulness and Tenderness (1-5) | Mixed model: score \~ lullaby + (1\|participant) + (1\|pair/song/snippet) |
| H1b | Secondary | Lullabies are rated lower on arousal | Arousal (1-9) | Same model |
| H1c | Secondary | Lullabies are not rated lower on valence (calm, not sad) | Valence (1-9), Sadness (GEMS) | Same model; equivalence test on valence |
| H2a | Secondary | Soothing presses rise after long held notes, especially plain (unornamented) ones | Press probability 0-3 s after elongation onsets | Event-locked average vs permutation baseline |
| H2b | Secondary | Presses rise after slow, wide glides | Same, glide onsets | Same |
| H2c | Secondary | Presses rise when percussion is quiet or absent | Press probability vs percussion level | Regression of press curve on features |
| H2d | Secondary | Presses rise after soft, gradual phrase onsets | Same, phrase onsets split by attack time | Same |
| H2e | Secondary | Presses fall during dense ornamentation (heavy kampita, fast turns) | Same, kampita onsets | Same |
| H3a | Secondary | Lullabies have a larger share of held notes | elongation.share\_of\_voiced\_time | Lullaby vs non-lullaby across all songs, effect size + permutation test |
| H3b | Secondary | Lullabies have a narrower pitch range and less time in the upper octave | range\_p5\_p95\_octaves, share\_tara | Same |
| H3c | Secondary | Lullabies have slower phrase onsets and a softer voice | phrase\_attack\_time\_ms, harmonic\_tilt\_db\_per\_oct | Same |
| H3d | Secondary | Lullabies have fewer, softer percussion strokes, quieter relative to the voice | stroke\_rate\_per\_s, perc\_to\_vocal\_db | Same |
| H3e | Secondary | Lullabies have fewer melodic turns per second | turning\_points\_per\_voiced\_s | Same |
| H4a | Exploratory | The H1 effect is present in every age group | Soothing score | lullaby x age group interaction |
| H4b | Exploratory | The H1 effect is stronger in older groups | Soothing score | Same, ordered contrast |

**Notes**

- Tempo is not in H3 yet: the current tempo estimates can be off by a factor of 2. Tempo enters H3 only after the beat of each song is checked by ear.
- H3 is tested on the snippets participants actually hear and, separately, on the whole songs.
- A first look at the whole-song features already leans towards H3a-c. That is why H3 must be written down before the remaining features are inspected.
- Familiarity, lyric understanding, musical training and mood are covariates, not hypotheses. Familiarity matters most: a listener who recognises a song as a lullaby may rate it by its label.

## Stimuli

Twenty recordings in 10 lullaby / non-lullaby pairs, labelled by the team. All 20 have complete features, processed with identical settings (BS-RoFormer on GPU, overlap 2, fp16; CREPE pitch). The mapping lives in `study/pairs.csv`.

### The 10 pairs

| Pair | Lullaby | Non-lullaby | Same singer | Suggested snippet (lullaby / non-lullaby) |
| --- | --- | --- | --- | --- |
| P1 | Alara Chanchalamaina (Annamayya, film) | Ye Divilo Virisina (Kanne Vayasu, film) | Not established | 16.5-54.0 s / 142.3-185.3 s |
| P2 | Sadiseyaku Gali (Raja Makutam, film) | Baa Beega Manamohana (P. Leela) | Not established | 135.7-174.8 s / 60.2-97.0 s |
| P3 | Morpeechhni Rajaai Odhi (Hariharan) | Shwaas Mee (Hariharan) | Yes | 333.6-370.2 s / 194.7-226.0 s |
| P4 | Laali Yugave | Idhigo Bhadradri (M. Balamuralikrishna) | Not established | 288.5-328.1 s / 91.0-128.9 s |
| P5 | Jo Achuthananda (M.S. Subbulakshmi) | Rara Chinnanna (M.S. Subbulakshmi) | Yes | 35.8-75.1 s / 64.7-102.4 s |
| P6 | Manikkam Katti (P. Unnikrishnan) | Radha Sametha Krishna (P. Unnikrishnan) | Yes | 271.2-307.7 s / 218.4-258.9 s |
| P7 | Omanathinkalkidavo | Paadunnu Njaaninnu | Not established | 517.1-560.1 s / 288.4-318.7 s |
| P8 | Tharattu paattu (Mangosteen Club) | Ayyappa song (K.J. Yesudas) | No | 8.6-48.8 s / 25.9-56.7 s |
| P9 | Palna Mein Lalna (Bombay Jayashri) | Deena Sharanyane (Bombay Jayashri) | Yes | 48.6-87.1 s / 67.2-101.0 s |
| P10 | Neej Mazya Nandlala (Lata Mangeshkar) | Aata Visavyache Kshana (Lata Mangeshkar) | Yes | 211.7-247.2 s / 69.9-113.7 s |

Five pairs (P3, P5, P6, P9, P10) share the singer and are the cleanest comparisons. The other five differ in singer and often in era or style; report results per pair and check that the overall effect does not rest on the mismatched pairs.

Open points on the set:

- **P8 lullaby:** the audio is a local file ("Tharattu paattu"); confirm it is the recording at the YouTube link the team chose (1WqAz1Gef3U).
- **P7 lullaby:** the recording is 15 minutes long; check by ear that the chosen section sounds like the lullaby.
- **Language and singer fields** marked "Not established" should be filled in by the team in `study/pairs.csv`.

### Labelling a lullaby

Write a one-paragraph definition before data collection, for example: a song whose text or traditional use is to put a child to sleep (functional lullaby), versus a song in lullaby style written for another purpose (e.g. a film scene). Two team members label every song independently; report their agreement and resolve disagreements by discussion. Film songs (P1, P2) need explicit decisions under this rule.

### Matching criteria, in priority order

1. Same singer.
2. Same language.
3. Similar era and recording quality (mono/stereo, studio/live, noise).
4. Similar instrumentation.
5. Similar vocal brightness and vocal-to-accompaniment level (from the features), so loudness or timbre does not stand in for "lullaby".

Record for each pair which criteria it meets; this becomes a table in the paper.

### Snippets

| Property | Current build | Recommendation |
| --- | --- | --- |
| Snippets per song | 1 | **2** non-overlapping windows; each participant hears one (see Design) |
| Duration | 30-45 s, varies by song | **40 s +/- 3 s** (37-43 s), mean equal in both conditions |
| Start / end | Just before a phrase begins / just after one ends (0.3 s margin) | Same |
| Selection rule | Window with the highest share of singing | Window closest to the whole song's feature profile (held-note share, range, glides, oscillations, percussion), among windows with at least 70 % singing |
| Loudness | -23 LUFS integrated (EBU R128, two-pass), true peak <= -2 dBTP | Same |
| Fades | 0.5 s in and out | Same |
| Format | MP3 192 kbit/s, 44.1 kHz stereo | Same |
| What is heard | Full mix (voice + accompaniment) | Same: listeners hear music as it is normally heard; features cover voice and percussion |
| File names | Random tokens, so a URL never reveals the condition | Same |

**Why 40 s.** Felt emotion takes roughly 20-30 s to build; below 30 s listeners mostly report the emotion they recognise. Above about 45 s the 10 snippets no longer fit a 35-minute session. A fixed length avoids "longer snippets" explaining a difference between conditions.

**Why representative windows.** Choosing the most-sung section favours dense passages that may not represent the song. Choosing the window closest to the whole song's profile, under a rule fixed before data collection, avoids cherry-picking and is easy to report.

**How snippets are made.** `python -m experiment.snippets suggest` writes proposed windows to `study/snippets.csv`; the team listens to each and may edit start and end; `python -m experiment.snippets cut` produces the audio. Both team members approve the final windows before any data is collected. P4 non-lullaby (Idhigo Bhadradri) currently has only 60 % singing in its window and needs a second look.

## Participants

Target: 120 completed sessions, 30 in each of four adult age groups; recruit about 140 to allow for exclusions.

| Age group | Target completers | Recruit |
| --- | --- | --- |
| 18-25 | 30 | \~35 |
| 26-40 | 30 | \~35 |
| 41-60 | 30 | \~35 |
| 61+ | 30 | \~35 |

The age group is derived from the age the participant enters, so it cannot be mis-selected. Under-18s are stopped at the profile page. Children are excluded from this study; a children's version needs parental consent, a separate ethics route and a simpler task, and is best run as its own study.

**Eligibility**

- 18 or older; self-reported normal hearing.
- Headphones or earphones (speakers are refused at the profile page; the headphone check confirms it).
- A quiet place and about 35 minutes.
- Indian listeners as the main population. Decide whether listeners without an Indian music background are excluded or form a separate comparison group (see Decisions).

**Sample size rationale.** 30 per group gives each snippet about 60 listeners with one snippet per song, or about 30 with two versions per song. About 30 listeners per snippet is enough for stable press-probability curves in H2. For H1 the limiting factor is the number of songs (10 pairs), not the number of listeners. The pilot's variability will be used for a simulation-based power check before the main study; adjust the target if needed.

**Recruitment.** Each channel (for example a WhatsApp group, a university list, a music school, a senior citizens' group) gets its own private link, so we know where each participant came from and can balance channels across age groups. Decide on compensation per channel (see Decisions).

**Information collected about each participant:** age, gender, first language, other languages understood, region where they grew up, years of Indian classical training, years of other music training, how often they listen to Indian classical or devotional music, whether lullabies were sung to them, whether they have sung lullabies to a child, listening device, and current mood (valence and arousal). No name, email or phone number is collected.

## Design and counterbalancing

Each participant hears 10 snippets: one song from every pair, five lullabies and five non-lullabies, and never both songs of a pair. Hearing both songs of a pair (often the same singer) would make the comparison obvious and invite guessing the hypothesis.

**Sets.** Set A holds the lullabies of P1-P5 and the non-lullabies of P6-P10; Set B holds the reverse. Every song is heard by half the participants.

| Set | Lullabies heard | Non-lullabies heard |
| --- | --- | --- |
| A | P1, P2, P3, P4, P5 | P6, P7, P8, P9, P10 |
| B | P6, P7, P8, P9, P10 | P1, P2, P3, P4, P5 |

**Balancing.** The server assigns the set with fewer participants in that person's age group (random on a tie), so A and B stay equal within every age group. Participants excluded by the headphone check do not count. In a test with 40 simulated sign-ups, every age group split 5 / 5.

**Versions (if two snippets per song are adopted).** Each song gets version 1 and version 2; the server balances versions the same way, so each version of each song is heard by about a quarter of participants. Snippet becomes a random effect inside song in the analysis.

**Order.** Phase 1 order is random per participant. Phase 2 uses a second, independent random order. A practice snippet from outside the set (a Carnatic recording, Mokshamu Galada) comes first in each phase and is not analysed.

**Attention check.** One random Phase 1 trial per participant carries an extra row: "Please choose 2 for this row."

**Session time budget**

| Part | Time |
| --- | --- |
| Consent, profile, mood | 4 min |
| Volume setting + headphone check | 3 min |
| Phase 1: practice + 10 snippets (40 s listening + 60-90 s rating each) | 20 min |
| Phase 2: practice + 10 snippets (40 s each + short pause) | 8 min |
| Final questions | 2 min |
| **Total** | **about 35-37 min** |

Each extra Phase 2 label (for example "tense or exciting") adds a full re-listen of about 8 minutes. Each extra pair per participant adds about 3 minutes; 10 snippets is the practical ceiling for unpaid online volunteers.

## Procedure

The whole session runs in one browser page. Every answer is saved to the server as soon as it is given; if the connection drops, answers wait in the browser and are sent later, and a reload continues from the same step.

1. **Link.** The participant opens a private study link (`<server>/?k=<key>`). An invalid link shows an error and nothing else.
2. **Welcome and consent.** A short description (10 pieces, \~40 s each, \~35 minutes, headphones and a quiet place) and the ethics-approved consent text. Two boxes must be ticked: "I am 18 or older and agree to take part" and "I have headphones and I am in a quiet place".
3. **Profile.** Age, gender, languages, region, music training, listening habits, lullaby experience, listening device, and current mood (two 1-9 scales). Missing answers are highlighted; choosing "Speakers" stops the participant. On submit the server assigns age group, set and orders.
4. **Volume.** The participant plays a 12 s music sample and sets a comfortable level, then is asked not to change it.
5. **Headphone check** (Woods et al., 2017). Six trials of three 200 Hz tones; one is 6 dB quieter, one is played in antiphase between the ears. Over loudspeakers the antiphase tone cancels and sounds quietest; over headphones the truly quieter tone does. The participant picks the quietest tone. Pass = 5 of 6 correct; two attempts; a reload does not add attempts. Failing ends the session politely and frees the slot for balancing.
6. **Phase 1 instructions.** "Tell us how the music made YOU feel while listening, not what the music is meant to express."
7. **Phase 1 practice** (one snippet, not analysed).
8. **Phase 1 trials (10).** Press Play; the snippet plays once with no pause, seek or replay; the rating form appears only after it ends. Submit is blocked until every scale is answered.
9. **Phase 2 instructions.** "Press and HOLD the button (or the space bar) whenever the music makes you feel calm or soothing. Release it when the feeling stops. You can press as often as you like, or not at all."
10. **Phase 2 practice** (one snippet).
11. **Phase 2 trials (10)** in a new random order. A large button turns green while held; the space bar works on laptops. After each snippet the participant sees how many moments they marked.
12. **Final questions.** What they think the study was about; whether they were in a quiet place; anything that went wrong.
13. **Completion.** A completion code is shown and stored; the page cannot be restarted from the same browser.

**In-person pilot.** 5-8 people do the online session on a lab laptop with a researcher present, then a short interview: were instructions clear, which scales were confusing, was anything too long, did the hold task feel natural. Time each part. Adjust and freeze the materials before the main study.

## Measures

### Phase 1: after each snippet

| Measure | Scale | Anchors | Use |
| --- | --- | --- | --- |
| GEMS-9: Wonder, Transcendence, Tenderness, Nostalgia, Peacefulness, Power, Joyful activation, Tension, Sadness | 1-5 each | not at all ... very much (each with three describing words) | H1; soothing score = mean(Peacefulness, Tenderness) |
| Valence | 1-9 | very unpleasant ... very pleasant | H1c |
| Arousal | 1-9 | very calm, sleepy ... very excited, alert | H1b |
| Liking | 1-5 | not at all ... very much | Covariate / secondary |
| Familiarity | 4 options | No / Maybe / Yes / Yes, I know it well | Covariate; sensitivity analysis |
| Lyric understanding | 3 options | All or most / Some / None | Covariate; sensitivity analysis |
| Free words | Up to 3 words, optional | e.g. peaceful, homesick, sleepy | Qualitative; coded into GEMS categories |
| Response time | seconds | from the end of the audio to submit | Quality control |

GEMS-9 is the short Geneva Emotional Music Scale, designed for emotions felt in response to music. It replaces the original plan of asking participants to list 5-10 emotion words, which would give answers that cannot be compared across people. The optional free words keep the open-ended material.

### Phase 2: during each snippet

For every hold: start and end time in snippet seconds, measured on the audio clock (`AudioContext.currentTime`), so the times refer to what was actually played. Stored per trial: number of holds, the device's reported output latency (typically 10-40 ms; more over Bluetooth), input type (touch or mouse/keyboard), and whether the tab was hidden while the music played.

Per snippet these holds become a press-probability curve: the share of listeners holding at each 100 ms.

### Questionnaires

- **Before:** the profile (see Participants) and current mood.
- **Music background:** years of Indian classical training, years of other training, listening frequency. Optionally add the Gold-MSI Musical Training subscale (7 items) from its official source if the team wants a standard score.
- **After:** what the study was about (a check for hypothesis guessing), listening conditions, problems.

## Data collection software and hosting

The experiment app is built, tested end to end in a browser on desktop and phone widths, and committed to the repository (`experiment/`, guide in `docs/STUDY_APP.md`).

| Part | What it does | Where |
| --- | --- | --- |
| Participant app | All steps above in one page; plain JavaScript, no external libraries; works on laptops and phones | `experiment/static/` |
| Server | Validates links, assigns age group / set / order, serves snippets by random token, stores answers | `experiment/server.py` (FastAPI) |
| Storage | One SQLite database; answers keyed by participant, kind and item, so retries never duplicate | `study/data/responses.sqlite` |
| Links | One private random key per recruitment channel | `python -m experiment.links add <label>` |
| Snippet tool | Suggests windows, cuts and loudness-normalises the audio | `python -m experiment.snippets suggest` / `cut` |
| Export | Tidy CSV files for analysis | `python -m experiment.export` |
| Live counts | Participants per age group, set and status | `/admin/status` with an admin token |

**What participants never see:** which snippet is a lullaby (file names are random tokens; the mapping stays on the server), the pair structure, or the hypotheses.

**What never goes to GitHub:** the snippet audio (copyrighted) and everything under `study/data/` (responses and links). Both are excluded by `.gitignore`; the repository is public.

### Hosting

| Stage | Option | Cost | Notes |
| --- | --- | --- | --- |
| Pilot | Cloudflare quick tunnel from a laptop: `./scripts/start_study.sh --public` | Free | URL changes every run; the laptop must stay on and awake |
| Main study (recommended) | Institution server with HTTPS | Free | Data stays on a machine the institution controls, which ethics committees usually prefer; needs IT to open a port or reverse proxy |
| Main study (alternative) | Small cloud VM (e.g. 1 vCPU, 1 GB) with HTTPS and a fixed domain | \~USD 5-10 / month | Quick to set up; someone must keep it running |

On any server: run `python -m experiment.server` with `HOST=0.0.0.0`, copy `study/snippets/` and `study/data/links.json` there, set `STUDY_ADMIN_TOKEN`, and back up `responses.sqlite` daily to a second location.

### Built-in decisions that can be changed

- Phones are allowed (press-and-hold button) and the device type is recorded; switch to desktop only if timing precision matters more than reach.
- The interface is English only; texts live in `study/config.json` and can be translated.
- The consent text and contact in `study/config.json` are placeholders and must be replaced before launch.

## Data collected

`python -m experiment.export` writes four CSV files to `study/data/export/`.

| File | One row per | Main columns |
| --- | --- | --- |
| `participants.csv` | participant | pid, created / finished time, status (complete / started / excluded), completion code, link label, age group, set, every profile answer, device (touch, screen size, audio sample rate), headphone result (correct, passed), attention trial |
| `ratings.csv` | participant x snippet | pid, token, pair, condition (lullaby / non\_lullaby / practice), track, snippet start / end in the song, 9 GEMS ratings, valence, arousal, liking, familiar, lyrics, words, response time, position in order, attention answer, tab hidden during audio |
| `presses.csv` | hold | pid, token, pair, condition, track, snippet start / end, label, hold number, start\_s, end\_s, duration\_s (snippet seconds), output latency |
| `events.csv` | event | volume step, final questions (study guess, quiet place, problems) |

**Linking presses to features.** A press at `start_s` in a snippet corresponds to `snippet_start_s + start_s` in the song, which indexes the song's `timeseries.csv` (10 ms frames of pitch, voicing, events, elongations, loudness, percussion level, local tempo).

## Quality control and exclusion rules

All rules are fixed in the pre-registration before data collection and applied by script, not by judgement.

**Participant-level exclusions**

| Rule | Threshold | Status |
| --- | --- | --- |
| Headphone check failed | fewer than 5 of 6 correct on both attempts | Built in (status = excluded) |
| Attention check failed | answer other than 2 on the check row | Applied at analysis |
| Straight-lining | the same value on more than 80 % of rating rows across all trials | Applied at analysis |
| Too fast | total time under 50 % of the median completion time | Applied at analysis |
| Did not finish | status not complete | Applied at analysis (partial data kept for reporting only) |
| Under 18 | age entered below 18 | Built in (cannot start) |

**Trial-level exclusions**

- The tab was hidden while the music played (`tab_hidden_during_audio > 0`).
- Practice trials (condition = practice).
- Phase 2: an implausible number of holds (for example more than 40 in one snippet); threshold to be set from the pilot.

**Reliability checks**

- Agreement between listeners per scale (intraclass correlation over snippets). Scales with very low agreement are reported but not interpreted.
- Split-half reliability of the press-probability curves (random halves of listeners per snippet).
- Two-rater agreement on the lullaby labels and on coding the free words.

## Analysis plan

### H1: do lullabies feel more soothing?

**Primary outcome:** soothing score = mean of GEMS Peacefulness and Tenderness. One primary outcome avoids testing 12 scales at once.

**Primary model** (linear mixed-effects, `lme4` in R or `statsmodels` / `pymer4` in Python):

```latex
\text{soothing}_{ij} = \beta_0 + \beta_1\,\text{lullaby}_j + u_i + v_{\text{pair}(j)} + w_{\text{song}(j)} + \varepsilon_{ij}
```

with random intercepts for participant i, pair, and song j within pair (add snippet within song if two versions are used). H1a is supported if beta\_1 > 0 at alpha = 0.05, two-sided. Report beta\_1 with its 95 % confidence interval and a standardised effect size.

**Secondary outcomes:** arousal (expect beta\_1 < 0), valence (equivalence test: the difference lies within +/- 0.5 scale points), each GEMS scale separately with Holm correction.

**Simple check:** the lullaby minus non-lullaby difference of mean soothing score per pair, with a sign test over the 10 pairs (how many go the predicted way).

**Sensitivity analyses**

1. Add familiarity and lyric understanding as covariates.
2. Exclude trials where the listener knew the song well.
3. Use only the five same-singer pairs.
4. Add musical training and starting mood as covariates.

### H2: which moments soothe?

1. **Press curves.** For each snippet, the share of listeners holding the button at every 100 ms; smooth lightly (e.g. 300 ms window).
2. **Event-locked averages.** For each feature event (elongation onset, plain vs ornamented elongation, glide onset, kampita onset, sphurita, percussion entry, phrase onset), average the change in press probability 0-3 s after the event against 1 s before it. Compare with the same measure at random times in the same snippets (1,000 permutations).
3. **Regression on features.** Model the press curve with the feature time series (voicing, held-note indicator, pitch velocity, oscillation indicator, percussion level, vocal loudness, local tempo), each lagged by 0.5-2 s. Use a mixed model with snippet as a random effect, or a generalised additive model. The lag that fits best estimates the listener response time.
4. **Latency.** Subtract each trial's reported output latency before alignment; check that results hold without it.

### H3: do lullabies sound different?

Compare lullabies with non-lullabies on the pre-registered features (held-note share, p5-p95 range, upper-octave share, phrase attack time, harmonic tilt, stroke rate, percussion-to-vocal level, turning points per second). With 10 per group: report effect sizes (Hedges' g) with bootstrap confidence intervals and a paired permutation test over the 10 pairs. Run on the snippets and on the whole songs. Present as descriptive support, not as the main result.

### Linking H1 and H3

Do the acoustic differences explain the felt differences? Add the snippet's features as predictors to the H1 model one at a time (or as a few principal components) and see how much of the lullaby effect they absorb. With 20 snippets this is exploratory.

### H4: does age matter?

Add the lullaby x age group interaction to the H1 model; test it with a likelihood-ratio test. Report the lullaby effect within each age group with confidence intervals. Explore lullabies heard in childhood and lullabies sung to a child as moderators.

### Free words

Two coders independently map each word to a GEMS category (or "other"); report Cohen's kappa. Compare category counts between lullabies and non-lullabies; show the most frequent words per condition.

### Reporting

Deviations from the pre-registration are listed in the paper. Data (without free-text answers that could identify someone) and analysis code are shared; snippet audio is not, for copyright reasons, but the windows and YouTube IDs are.

## Where the extracted MIR features are used

The pipeline's features enter the study at six points, from choosing the stimuli to explaining the results.

| Stage | Features | Purpose |
| --- | --- | --- |
| Choosing and checking pairs | Vocal brightness, harmonic tilt, vocal-to-accompaniment level, recording quality | Pairs should differ mainly in being a lullaby, not in timbre or mix |
| Cutting snippets | Voicing and phrase boundaries; summary profile per song | Windows that start and end between phrases and represent the whole song |
| H3 | Held-note share, range, upper-octave share, glides, oscillations (simple vs heavy kampita), sphurita, phrase attack, harmonic tilt, percussion stroke rate and level, melodic turning points | Do lullabies sound different? |
| H2 | Time-stamped events in each song's `timeseries.csv` (10 ms): elongations, glides, kampita, sphurita, loudness, percussion level, local tempo | Which moments are followed by soothing presses |
| Linking H1 and H3 | Per-snippet feature summaries | Do acoustic differences explain felt differences? |
| Controls | Tempo, loudness, brightness | Rule out "it is just slower, quieter or darker" |

**Feature definitions** (all thresholds are saved in every `features.json`):

- **Elongation:** a held svara of at least 0.6 s, plain (stable pitch) or ornamented (oscillation centred on one svara).
- **Glide:** monotonic pitch movement of at least 150 cents taking at least 80 ms.
- **Sphurita:** a steady svara, a dip of 60-350 cents lasting up to 150 ms, then back.
- **Oscillation:** at least 4 alternating turning points (swings of at least 30 cents, 40-350 ms apart); simple below 150 cents peak to peak, heavy kampita above.
- **Range:** p5-p95 and p1-p99 in cents and octaves; share of time in mandra, madhya and tara sthayi.
- **Tempo:** local tempo from percussion onsets (verify by ear; can be off by a factor of 2).
- **Vocal softness:** phrase attack time and slope, harmonic tilt, alpha ratio, H1-H2, loudness range, vocal-to-accompaniment level.
- **Percussion softness:** per stroke attack time, brightness, crest factor and level; stroke rate; percussion-to-vocal level.

**Known limits to state in the paper:** vocal separation is trained on Western music, so violin and tanpura can leak into the vocal stem; CREPE under-reads fast, wide kampita by about 10-15 %; stroke counts in very soft percussion are unreliable.

## Decisions to make

The first group blocks snippet cutting and must be settled first; the rest can follow the pilot. Fill in the Decision column as the team agrees.

### Before cutting snippets

| # | Question | Options | Recommended default | Decision |
| --- | --- | --- | --- | --- |
| D1 | Snippets per song | 1 / 2 | 2, one per participant, balanced |  |
| D2 | Snippet length | 30-45 s variable / fixed | 40 s +/- 3 s, equal mean across conditions |  |
| D3 | Window selection rule | most singing / most representative | Closest to the whole-song feature profile, at least 70 % singing |  |
| D4 | Felt or perceived emotion | felt / perceived | Felt (as built) |  |
| D5 | Primary outcome | one scale / composite / all scales | Soothing = mean(Peacefulness, Tenderness) |  |
| D6 | Phase 2 labels | calm or soothing / + tense or exciting | Calm or soothing only |  |
| D7 | Full mix or vocals only | mix / vocals | Full mix |  |

### Before the pilot

| # | Question | Options | Recommended default | Decision |
| --- | --- | --- | --- | --- |
| D8 | Lullaby definition | written rule | Functional lullaby vs lullaby-style song, two independent raters |  |
| D9 | P8 recording | local file / re-download | Confirm the local file matches link 1WqAz1Gef3U |  |
| D10 | Rating items per snippet | GEMS-9 + valence + arousal + liking / shorter | Keep; cut to 6 GEMS items if the pilot shows fatigue |  |
| D11 | Free emotion words | none / optional up to 3 / required 5-10 | Optional, up to 3 |  |
| D12 | Phones | allowed / desktop only | Allowed, device recorded |  |
| D13 | Interface language | English / + Hindi / + Telugu / others | English; add others if recruiting older or non-English-speaking groups |  |
| D14 | Gold-MSI | add Musical Training subscale / own questions only | Own questions; add Gold-MSI if reviewers in the target venue expect it |  |

### Before the main study

| # | Question | Options | Recommended default | Decision |
| --- | --- | --- | --- | --- |
| D15 | Age groups | as below / other cut points | 18-25, 26-40, 41-60, 61+ |  |
| D16 | Sample size | fixed / from pilot simulation | 30 completers per group, checked by simulation |  |
| D17 | Population | Indian listeners only / + non-Indian group | Indian listeners; a non-Indian group only as a separate, later study |  |
| D18 | Recruitment and compensation | per channel | One link per channel; compensation as ethics allows |  |
| D19 | Hosting | institution server / cloud VM | Institution server with HTTPS |  |
| D20 | Exclusion thresholds | as in Quality control | As written; Phase 2 hold limit from the pilot |  |
| D21 | Pre-registration venue | OSF / AsPredicted | OSF |  |

## Risks and mitigations

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Pairs differ in more than lullaby status | H1 effect attributable to singer, era or recording | Matching criteria; same-singer subset as sensitivity analysis; report per pair |
| Listeners recognise lullabies by words or fame | Ratings reflect the label, not the music | Measure familiarity and lyric understanding; sensitivity analyses without known songs |
| Unrepresentative snippet | One section drives a pair's result | Two versions per song; selection rule fixed in advance |
| Online listening conditions vary | Noise in ratings and press timing | Headphone check, volume step, tab-visibility log, latency stored per trial |
| Hypothesis guessing | Inflated H1 | Never both songs of a pair; neutral study title; "what was this about?" question |
| Fatigue in later trials | Lower data quality at the end | 35-minute cap; random order; position stored and modelled |
| Too few older participants online | Weak H4 | Dedicated channels for 41-60 and 61+; option of assisted in-person sessions |
| Separation artefacts in features | H2 / H3 noise | State limits; check key features by ear for the 20 snippets |
| Server down during collection | Lost sessions | Answers queue in the browser and retry; daily backups; monitor `/admin/status` |

### Ethics, copyright and data protection

- **Ethics approval** before any data collection, including the pilot. The consent text in `study/config.json` must be the approved one.
- **Consent** covers: purpose in general terms (music and feelings), duration, voluntariness, right to stop at any time, anonymous storage, research use only, contact person.
- **Personal data.** No names or contact details are collected; participants get a random ID. Age, languages and region are stored. Free-text answers are checked for identifying details before any data is shared.
- **Storage.** Responses stay on a server the team controls, are backed up daily, and are never committed to the public repository. Decide and state a retention period.
- **Copyright.** Snippets are short excerpts used for research only; they are served only to participants through private links and are never published or committed. Papers cite the recordings (title, artist, YouTube ID) and the snippet windows.

## Timeline and next steps

No dates are fixed yet; the order below is the critical path. Steps 1-4 can run in parallel with the ethics application.

1. **Settle decisions D1-D7** (team meeting).
2. **Finalise snippets:** apply the selection rule, listen to every window, both team members approve, cut the audio.
3. **Confirm stimulus details:** P8 recording, lullaby definition and two-rater labels, language and singer fields in `study/pairs.csv`.
4. **Update the app** for any decisions that change it (two versions per song, fixed length, labels, languages); replace the consent text and contact.
5. **Ethics approval.**
6. **Pilot** with 5-8 people in person; interview; time each part.
7. **Revise** instructions, scales and thresholds from the pilot; freeze the materials.
8. **Pre-register** H1-H4, the primary outcome, exclusion rules, models and sample size on OSF.
9. **Set up hosting** (institution server with HTTPS), private links per channel, daily backups.
10. **Main data collection** until about 30 completers per age group; monitor balance weekly.
11. **Export, apply exclusions, analyse** as pre-registered; then exploratory analyses.
12. **Write up.**

**Done already:** feature extraction for all 20 songs; the pairs mapping; suggested snippet windows and loudness-normalised audio; the experiment app with assignment, storage, export and private links, tested end to end.
