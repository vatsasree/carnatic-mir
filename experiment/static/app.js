// Listening study – participant app. One page; every step saves to the server as soon as it is done.
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const APP = $('#app');
const K = new URLSearchParams(location.search).get('k') || '';
const LS = 'study:' + K;
let CFG, S = {};                 // S = persistent participant state {pid, assignment, step, ...}
let ctx;                         // AudioContext (created on first click)
const buffers = new Map();

// ---------- persistence ----------
const load = () => { try { return JSON.parse(localStorage.getItem(LS)) || {}; } catch { return {}; } };
const persist = () => { try { localStorage.setItem(LS, JSON.stringify(S)); } catch { } };
const outboxKey = LS + ':outbox';
const getOutbox = () => { try { return JSON.parse(localStorage.getItem(outboxKey)) || []; } catch { return []; } };
const setOutbox = q => { try { localStorage.setItem(outboxKey, JSON.stringify(q)); } catch { } };

async function api(path, body, tries = 4) {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(path, body === undefined ? {} :
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (r.ok) return r.json();
      const msg = (await r.json().catch(() => ({}))).detail || r.statusText;
      if (r.status < 500) throw Object.assign(new Error(msg), { fatal: true });
      throw new Error(msg);
    } catch (e) {
      if (e.fatal || i >= tries - 1) throw e;
      await sleep(600 * 2 ** i);
    }
  }
}
// Saves never block the participant: queued locally, retried until the server confirms.
function save(kind, item, payload) {
  const q = getOutbox(); q.push({ pid: S.pid, kind, item, payload, client_ts: Date.now() / 1000 }); setOutbox(q);
  flush();
}
let flushing = null;
async function drain() {
  let q = getOutbox();
  while (q.length) {
    try { await api('/api/save', q[0], 3); }
    catch (e) { if (!e.fatal) return false; }        // offline: keep the queue, try again later
    q = getOutbox(); q.shift(); setOutbox(q);         // saved (or rejected for good): drop it
  }
  return true;
}
function flush() {
  if (!flushing) flushing = drain().finally(() => { flushing = null; });
  return flushing;
}
setInterval(() => { if (getOutbox().length) flush(); }, 10000);
const sleep = ms => new Promise(r => setTimeout(r, ms));
function toast(msg, ms = 2500) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => t.hidden = true, ms); }
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function screen(html) { APP.innerHTML = html; window.scrollTo(0, 0); return APP; }
function next(label = 'Continue') { return `<div class="actions"><button id="next">${label}</button></div>`; }
const clicked = el => new Promise(r => el.addEventListener('click', r, { once: true }));

// ---------- page-visibility log (audio steps note if the tab was hidden) ----------
let hiddenLog = null;
document.addEventListener('visibilitychange', () => { if (hiddenLog && document.hidden) hiddenLog.push(performance.now()); });

// ---------- audio ----------
function audioCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}
async function buffer(url) {
  if (!buffers.has(url)) buffers.set(url, (async () => {
    const r = await fetch(url); if (!r.ok) throw new Error('audio ' + r.status);
    const a = await r.arrayBuffer();
    return await new Promise((res, rej) => audioCtx().decodeAudioData(a, res, rej));
  })());
  return buffers.get(url);
}
// Plays a buffer once; returns {t0, done} where t0 is the AudioContext time at which playback started.
function play(buf, onProgress) {
  const c = audioCtx(), src = c.createBufferSource();
  src.buffer = buf; src.connect(c.destination);
  const t0 = c.currentTime + 0.05; src.start(t0);
  let raf;
  const tick = () => { onProgress && onProgress(Math.min(1, (c.currentTime - t0) / buf.duration)); raf = requestAnimationFrame(tick); };
  tick();
  const done = new Promise(r => src.onended = () => { cancelAnimationFrame(raf); onProgress && onProgress(1); r(); });
  return { t0, done, stop: () => src.stop() };
}
const url = tok => `/audio/${tok}.mp3`;

// ---------- form widgets ----------
function pills(name, n, left, right, q, hint = '') {
  const cols = `grid-template-columns: repeat(${n}, 1fr)`;
  return `<div class="scale" data-name="${name}"><div class="q">${q}</div>${hint ? `<div class="hint">${hint}</div>` : ''}
    <div class="pills" style="${cols}">${Array.from({ length: n }, (_, i) =>
      `<label><input type="radio" name="${name}" value="${i + 1}" aria-label="${esc(q)} ${i + 1}"><span>${i + 1}</span></label>`).join('')}</div>
    <div class="ends"><span>${left}</span><span>${right}</span></div></div>`;
}
function options(name, q, opts) {
  return `<div class="scale" data-name="${name}"><div class="q">${q}</div><div class="opts">${opts.map(([v, l]) =>
    `<label><input type="radio" name="${name}" value="${v}">${l}</label>`).join('')}</div></div>`;
}
function readForm(root) {
  const out = {}, missing = [];
  $$('.scale', root).forEach(s => {
    const n = s.dataset.name, v = $(`input[name="${n}"]:checked`, s);
    s.classList.toggle('missing', !v);
    if (v) out[n] = isNaN(+v.value) ? v.value : +v.value; else missing.push(s);
  });
  return { out, missing };
}

// ---------- steps ----------
async function welcome() {
  const t = CFG.text;
  screen(`<h1>${esc(CFG.title)}</h1><div class="card"><p>${esc(t.welcome)}</p></div>
    <h2>Consent</h2><div class="card"><p>${esc(t.consent)}</p><p class="small muted">Contact: ${esc(CFG.contact)}</p>
    <label class="check"><input type="checkbox" id="c1"> I am 18 or older and agree to take part.</label>
    <label class="check"><input type="checkbox" id="c2"> I have headphones and I am in a quiet place.</label></div>
    ${next('Start')}`);
  const b = $('#next'); b.disabled = true;
  const upd = () => b.disabled = !($('#c1').checked && $('#c2').checked);
  $('#c1').onchange = $('#c2').onchange = upd;
  await clicked(b);
  audioCtx();
}

const LANGS = ['Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Hindi', 'Marathi', 'Gujarati', 'Bengali', 'Sanskrit', 'English'];
async function profile() {
  screen(`<h1>About you</h1><div class="card" id="f">
    <label class="field" for="age">Age (years)</label><input id="age" type="number" min="18" max="120" inputmode="numeric">
    ${options('gender', 'Gender', [['woman', 'Woman'], ['man', 'Man'], ['other', 'Another gender'], ['na', 'Prefer not to say']])}
    <label class="field" for="lang1">First language (mother tongue)</label>
    <select id="lang1"><option value="">— choose —</option>${LANGS.map(l => `<option>${l}</option>`).join('')}<option value="other">Other</option></select>
    <input id="lang1o" type="text" placeholder="Please specify" hidden style="margin-top:8px">
    <label class="field">Other languages you understand</label>
    <div class="chips" id="langs">${LANGS.map(l => `<label><input type="checkbox" value="${l}">${l}</label>`).join('')}</div>
    <input id="langso" type="text" placeholder="Others (comma separated)" style="margin-top:8px">
    <label class="field" for="region">State / region where you grew up</label><input id="region" type="text">
    <label class="field" for="yrsIndian">Years of training in Indian classical music (singing or instrument)</label>
    <input id="yrsIndian" type="number" min="0" max="80" inputmode="numeric" value="">
    <label class="field" for="yrsOther">Years of training in any other music</label>
    <input id="yrsOther" type="number" min="0" max="80" inputmode="numeric" value="">
    ${options('listen', 'How often do you listen to Indian classical or devotional music?', [['never', 'Never'], ['rarely', 'A few times a year'], ['monthly', 'About once a month'], ['weekly', 'Every week'], ['daily', 'Almost every day']])}
    ${options('lullaby_heard', 'Were lullabies sung to you as a child?', [['yes', 'Yes'], ['no', 'No'], ['unsure', 'Not sure']])}
    ${options('lullaby_sung', 'Have you ever sung lullabies to a child?', [['yes', 'Yes'], ['no', 'No']])}
    ${options('headphones', 'What are you listening with?', [['over_ear', 'Over-ear or on-ear headphones'], ['in_ear', 'In-ear earphones / earbuds'], ['speakers', 'Speakers (no headphones)']])}
    <h2>Right now, how do you feel?</h2>
    ${pills('mood_valence', 9, 'very unpleasant', 'very pleasant', 'Unpleasant – pleasant')}
    ${pills('mood_arousal', 9, 'very calm, sleepy', 'very excited, alert', 'Calm – excited')}
    <p class="err" id="msg"></p></div>${next()}`);
  $('#lang1').onchange = e => $('#lang1o').hidden = e.target.value !== 'other';
  for (;;) {
    await clicked($('#next'));
    const { out, missing } = readForm($('#f'));
    const age = parseInt($('#age').value, 10);
    const lang1 = $('#lang1').value === 'other' ? $('#lang1o').value.trim() : $('#lang1').value;
    let err = '';
    if (!(age >= 18 && age <= 120)) err = 'Please enter your age (18 or older).';
    else if (!lang1) err = 'Please choose your first language.';
    else if (missing.length) { err = 'Please answer the questions marked in red.'; missing[0].scrollIntoView({ behavior: 'smooth', block: 'center' }); }
    else if (out.headphones === 'speakers') err = 'This study needs headphones or earphones. Please connect them and choose the right option.';
    if (err) { $('#msg').textContent = err; continue; }
    const prof = {
      ...out, age, lang1, region: $('#region').value.trim(),
      langs: [...$$('#langs input:checked').map(i => i.value), ...$('#langso').value.split(',').map(s => s.trim()).filter(Boolean)],
      years_indian_training: +$('#yrsIndian').value || 0, years_other_training: +$('#yrsOther').value || 0,
    };
    const device = { touch: 'ontouchstart' in window || navigator.maxTouchPoints > 0, w: window.screen.width, h: window.screen.height,
                     sample_rate: audioCtx().sampleRate };
    try {
      const r = await api('/api/start', { k: K, profile: prof, device });
      S.pid = r.pid; S.assignment = r.assignment; S.age_group = r.age_group; persist();
      return;
    } catch (e) { $('#msg').textContent = e.message; }
  }
}

async function calibrate() {
  screen(`<h1>Set your volume</h1><div class="card"><p>Put on your headphones. Press <b>Play sample</b> and set your
    volume to a comfortable level, as you would for listening to music. <b>Please do not change the volume after this.</b></p>
    <div class="actions"><button id="ps" class="secondary">Play sample</button><span class="muted small" id="st"></span></div></div>
    ${next('The volume is comfortable')}`);
  const b = $('#next'); b.disabled = true;
  const buf = await buffer(url(CFG.practice.token));
  let player = null;
  $('#ps').onclick = () => {
    if (player) { player.stop(); player = null; $('#ps').textContent = 'Play sample'; return; }
    // 12 s from the middle of the practice piece
    const c = audioCtx(), src = c.createBufferSource(), seg = c.createBuffer(buf.numberOfChannels, Math.min(buf.length, 12 * buf.sampleRate), buf.sampleRate);
    const off = Math.floor(buf.length / 3);
    for (let ch = 0; ch < buf.numberOfChannels; ch++) seg.copyToChannel(buf.getChannelData(ch).subarray(off, off + seg.length), ch);
    src.buffer = seg; src.connect(c.destination); src.start();
    player = { stop: () => src.stop() }; src.onended = () => { player = null; const ps = $('#ps'); if (ps) ps.textContent = 'Play sample'; };
    $('#ps').textContent = 'Stop'; b.disabled = false;
  };
  await clicked(b);
  if (player) player.stop();
  save('calibration', 'volume', { done: true });
}

// Headphone check (Woods et al., 2017): three 200 Hz tones, one is 6 dB softer, one is played in antiphase
// between the ears. Over loudspeakers the antiphase tone cancels and sounds softest; over headphones it does not.
function tone(kind) {
  const c = audioCtx(), sr = c.sampleRate, n = sr, b = c.createBuffer(2, n, sr);
  const amp = kind === 'quiet' ? 0.5 * 0.35 : 0.35, L = b.getChannelData(0), R = b.getChannelData(1), ramp = 0.1 * sr;
  for (let i = 0; i < n; i++) {
    const env = Math.min(1, i / ramp, (n - 1 - i) / ramp);
    const v = amp * env * Math.sin(2 * Math.PI * 200 * i / sr);
    L[i] = v; R[i] = kind === 'anti' ? -v : v;
  }
  return b;
}
async function headphoneCheck() {
  const H = CFG.headphone_check;
  for (let attempt = (S.hp_attempts || 0) + 1; attempt <= H.attempts; attempt++) {
    S.hp_attempts = attempt; persist();          // a reload does not give extra attempts
    let correct = 0; const trials = [];
    for (let t = 0; t < H.trials; t++) {
      const kinds = ['std', 'quiet', 'anti'].sort(() => Math.random() - .5);
      screen(`<h1>Sound check ${attempt > 1 ? '(second try)' : ''}</h1><div class="card">
        <p>You will hear three tones. Which one is the <b>softest</b> (quietest)?</p><p class="muted small">Tone ${t + 1} of ${H.trials}. Each set plays once.</p>
        <div class="actions"><button id="pl">Play the tones</button></div>
        <div class="choices" style="margin-top:16px">${[1, 2, 3].map(i => `<button class="secondary ans" data-i="${i - 1}" disabled>${i}</button>`).join('')}</div></div>`);
      await clicked($('#pl')); $('#pl').disabled = true;
      const c = audioCtx(); let at = c.currentTime + 0.1;
      for (const k of kinds) { const s = c.createBufferSource(); s.buffer = tone(k); s.connect(c.destination); s.start(at); at += 1.5; }
      await sleep((at - c.currentTime) * 1000);
      $$('.ans').forEach(b => b.disabled = false);
      const pick = await new Promise(r => $$('.ans').forEach(b => b.onclick = () => r(+b.dataset.i)));
      const ok = kinds[pick] === 'quiet'; correct += ok;
      trials.push({ order: kinds, pick, ok });
    }
    const passed = correct >= H.pass;
    save('headphone', 'attempt' + attempt, { correct, trials, passed });
    if (passed) { save('headphone', 'result', { correct, passed: true, attempts: attempt }); return true; }
    if (attempt < H.attempts) {
      screen(`<h1>Let's try that once more</h1><div class="card"><p>Please make sure your headphones are on properly
        (left and right on the correct ears) and that you are in a quiet place.</p></div>${next('Try again')}`);
      await clicked($('#next'));
    } else {
      save('headphone', 'result', { correct, passed: false, attempts: attempt });
    }
  }
  await flush();
  screen(`<h1>Thank you</h1><div class="card"><p>Unfortunately the sound check suggests that the sound is not reaching
    you through headphones, which this study needs. Thank you very much for your time; you can close this page now.</p></div>`);
  S.ended = true; persist();
  return new Promise(() => { });       // stop here
}

function intro(title, text, note) {
  return async () => {
    screen(`<h1>${title}</h1><div class="card"><p>${esc(text)}</p>${note ? `<p class="muted">${esc(note)}</p>` : ''}</div>${next()}`);
    await clicked($('#next'));
  };
}

async function listen(tok, heading) {
  screen(`<h1>${heading}</h1><div class="card player"><p>Press <b>Play</b> and listen until the end. The piece plays once.</p>
    <div class="progress"><div id="pg"></div></div><div class="actions"><button id="pl">Play</button><span id="st" class="muted small"></span></div></div>
    <div id="after"></div>`);
  const bufP = buffer(url(tok));
  await clicked($('#pl')); $('#pl').disabled = true; $('#st').textContent = 'Loading…';
  const buf = await bufP; $('#st').textContent = 'Playing…';
  hiddenLog = [];
  const p = play(buf, f => $('#pg').style.width = (f * 100) + '%');
  await p.done;
  const hidden = hiddenLog.length; hiddenLog = null;
  $('#st').textContent = 'Finished';
  return { duration: buf.duration, hidden };
}

function ratingTrial(tok, idx, total, practice = false) {
  return async () => {
    const heading = practice ? 'Practice piece' : `Piece ${idx + 1} of ${total}`;
    const L = await listen(tok, heading);
    const attention = !practice && idx === S.assignment.attention_trial;
    const rows = CFG.gems.map(([k, name, hint]) => pills('gems_' + k, 5, 'not at all', 'very much', name, hint));
    if (attention) rows.splice(4, 0, pills('attention', 5, 'not at all', 'very much', 'Attention check', 'Please choose 2 for this row.'));
    $('#after').innerHTML = `<div class="card" id="f"><h2 style="margin-top:0">How did this music make you feel?</h2>
      ${rows.join('')}
      ${pills('valence', 9, 'very unpleasant', 'very pleasant', 'Unpleasant – pleasant')}
      ${pills('arousal', 9, 'very calm, sleepy', 'very excited, alert', 'Calm – excited')}
      ${pills('liking', 5, 'not at all', 'very much', 'How much did you like it?')}
      ${options('familiar', 'Have you heard this song before?', [['no', 'No'], ['maybe', 'Maybe / not sure'], ['yes', 'Yes'], ['know_well', 'Yes, I know it well']])}
      ${options('lyrics', 'Did you understand the words?', [['all', 'Yes, all or most'], ['some', 'Some of them'], ['none', 'No']])}
      <label class="field" for="words">Up to three words for how it made you feel (optional)</label>
      <input id="words" type="text" maxlength="120" placeholder="e.g. peaceful, homesick, sleepy">
      <p class="err" id="msg"></p></div>${next()}`;
    const t0 = performance.now();
    for (;;) {
      await clicked($('#next'));
      const { out, missing } = readForm($('#f'));
      if (missing.length) { $('#msg').textContent = 'Please answer the rows marked in red.'; missing[0].scrollIntoView({ behavior: 'smooth', block: 'center' }); continue; }
      save('rating', tok, { ...out, words: $('#words').value.trim(), rt_s: Math.round(performance.now() - t0) / 1000,
                            attention_trial: attention, practice, tab_hidden_during_audio: L.hidden, position: practice ? -1 : idx });
      return;
    }
  };
}

function holdTrial(tok, label, idx, total, practice = false) {
  return async () => {
    const heading = practice ? 'Practice piece' : `Piece ${idx + 1} of ${total}`;
    screen(`<h1>${heading}</h1><div class="card player">
      <p>Hold the button (or the <b>space bar</b>) whenever the music makes you feel <b>${esc(label)}</b>.</p>
      <div class="progress"><div id="pg"></div></div>
      <button id="hold" class="hold" disabled>Press <b>Play</b> first</button>
      <div class="actions"><button id="pl">Play</button><span id="st" class="muted small"></span></div></div>
      <div id="after"></div>`);
    const bufP = buffer(url(tok));
    await clicked($('#pl')); $('#pl').disabled = true; $('#st').textContent = 'Loading…';
    const buf = await bufP; $('#st').textContent = '';
    const c = audioCtx(), hold = $('#hold');
    const holds = []; let down = null, p, playing = true;
    const now = () => Math.max(0, Math.min(buf.duration, c.currentTime - p.t0));
    const on = () => { if (!playing || down !== null) return; down = now(); hold.classList.add('on'); hold.textContent = 'Holding…'; };
    const off = () => { if (down === null) return; holds.push([+down.toFixed(3), +now().toFixed(3)]); down = null; hold.classList.remove('on'); hold.textContent = 'Hold while you feel ' + label; };
    hold.disabled = false; hold.textContent = 'Hold while you feel ' + label;
    hold.addEventListener('pointerdown', e => { e.preventDefault(); hold.setPointerCapture(e.pointerId); on(); });
    ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(ev => hold.addEventListener(ev, off));
    hold.addEventListener('contextmenu', e => e.preventDefault());
    const kd = e => { if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) on(); } };
    const ku = e => { if (e.code === 'Space') { e.preventDefault(); off(); } };
    const blur = () => off();
    addEventListener('keydown', kd); addEventListener('keyup', ku); addEventListener('blur', blur);
    hiddenLog = [];
    p = play(buf, f => $('#pg').style.width = (f * 100) + '%');
    await p.done;
    off(); playing = false;
    removeEventListener('keydown', kd); removeEventListener('keyup', ku); removeEventListener('blur', blur);
    hold.disabled = true; hold.textContent = 'Finished';
    save('presses', `${tok}|${label}`, { holds, duration_s: buf.duration, practice, position: practice ? -1 : idx,
      output_latency_s: c.outputLatency || c.baseLatency || null, tab_hidden_during_audio: hiddenLog.length,
      input: matchMedia('(pointer: coarse)').matches ? 'touch' : 'mouse/keyboard' });
    hiddenLog = null;
    $('#after').innerHTML = `<p class="muted">${holds.length ? `You marked ${holds.length} moment${holds.length > 1 ? 's' : ''}.` : 'You did not mark any moment – that is fine.'}</p>${next()}`;
    await clicked($('#next'));
  };
}

async function final() {
  screen(`<h1>Almost done</h1><div class="card" id="f">
    <label class="field" for="about">What do you think this study was about?</label><textarea id="about"></textarea>
    ${options('quiet', 'Were you in a quiet place without interruptions?', [['yes', 'Yes'], ['mostly', 'Mostly'], ['no', 'No']])}
    <label class="field" for="problems">Did anything go wrong (sound problems, interruptions)? (optional)</label><textarea id="problems"></textarea>
    <p class="err" id="msg"></p></div>${next('Finish')}`);
  for (;;) {
    await clicked($('#next'));
    const { out, missing } = readForm($('#f'));
    if (missing.length) { $('#msg').textContent = 'Please answer the question marked in red.'; continue; }
    save('final', 'questions', { ...out, about: $('#about').value.trim(), problems: $('#problems').value.trim() });
    return;
  }
}

async function done() {
  screen(`<h1>Saving…</h1><div class="card"><p>Please keep this page open for a moment.</p><p class="err" id="msg"></p></div>`);
  for (;;) {
    const ok = await flush();
    if (ok) try { const r = await api('/api/finish?pid=' + S.pid, {}); S.code = r.completion_code; S.ended = true; persist(); break; } catch { }
    $('#msg').textContent = 'Could not reach the server. Retrying… (check your internet connection)';
    await sleep(4000);
  }
  screen(`<h1>${esc(CFG.text.thanks)}</h1><div class="card"><p>Your completion code:</p><p class="code">${esc(S.code)}</p>
    <p class="muted small">You can close this page now.</p></div>`);
}

// ---------- flow ----------
function steps() {
  const a = S.assignment, n1 = a.phase1.length, n2 = a.phase2.length, t = CFG.text, pr = CFG.practice.token;
  const st = [calibrate, headphoneCheck,
    intro('Part 1: How does the music make you feel?', t.phase1_intro, t.practice_note),
    ratingTrial(pr, 0, n1, true),
    intro('Now the real pieces', 'The practice is over. Now the 10 pieces of the study follow.'),
    ...a.phase1.map((tok, i) => ratingTrial(tok, i, n1))];
  for (const label of CFG.phase2_labels) {
    st.push(intro('Part 2: Mark the moments', t.phase2_intro.replace('{label}', label), t.practice_note),
      holdTrial(pr, label, 0, n2, true),
      intro('Now the real pieces', 'The practice is over. Now the same 10 pieces follow, in a new order.'),
      ...a.phase2.map((tok, i) => holdTrial(tok, label, i, n2)));
  }
  st.push(final);
  return st;
}
const setBar = f => $('#barFill').style.width = (100 * f) + '%';

async function main() {
  if (!K) { screen('<h1>Link not valid</h1><p>Please use the full link you were given.</p>'); return; }
  try { CFG = await api('/api/config?k=' + encodeURIComponent(K)); }
  catch (e) { screen(`<h1>Link not valid</h1><p>${esc(e.message)}</p>`); return; }
  document.title = CFG.title; $('#title').textContent = CFG.title;
  S = load();
  if (S.ended) {
    screen(S.code ? `<h1>${esc(CFG.text.thanks)}</h1><div class="card"><p>Your completion code:</p><p class="code">${esc(S.code)}</p></div>`
      : '<h1>Thank you</h1><p>You have already taken part.</p>');
    return;
  }
  if (S.pid) {        // resume after a reload: the server's copy of the assignment wins
    try { const r = await api('/api/state?pid=' + S.pid); S.assignment = r.assignment; }
    catch { S = {}; }
  }
  flush();
  if (!S.pid) { await welcome(); await profile(); S.step = 0; persist(); }
  else {
    screen(`<h1>Welcome back</h1><div class="card"><p>You can continue where you stopped.</p></div>${next('Continue')}`);
    await clicked($('#next')); audioCtx();
  }
  const st = steps();
  for (let i = S.step || 0; i < st.length; i++) {
    setBar(i / st.length);
    await st[i]();
    S.step = i + 1; persist();
  }
  setBar(1);
  await done();
}
main().catch(e => { console.error(e); screen(`<h1>Something went wrong</h1><p class="err">${esc(e.message)}</p><p>Please reload the page; your answers so far are saved.</p>`); });
