/* SST web UI — transcribe, karaoke playback, live transcript editing */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

/* Current transcript being shown/edited */
const current = { jobId: null, filename: "", result: null, editingIdx: null, loop: null };
let lastJobs = [];
const isActive = (job) => job.status === "running" || job.status === "queued";

/* ---------------- tabs ---------------- */
$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".nav-item").forEach((b) => b.classList.remove("active"));
    $$(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "models") refreshModels();
    if (btn.dataset.tab === "dashboard") { refreshStatus(); refreshNetwork(); refreshLogs(); }
  });
});

/* ---------------- status ---------------- */
async function refreshStatus() {
  try {
    const r0 = await fetch("/api/status");
    if (r0.status === 401) { window.location.href = "/login?next=" + encodeURIComponent(location.pathname); return null; }
    const s = await r0.json();
    $("#server-dot").className = "dot ok";
    $("#server-label").textContent = "Server running";
    $("#device-label").textContent = s.device_description;
    $("#d-server").textContent = "Running · v" + s.version;
    $("#d-device").textContent = s.device_description;
    $("#d-stt").textContent = s.stt_loaded || "not loaded yet";
    $("#d-diar").textContent = s.diarization_loaded || "not loaded yet";
    $("#ffmpeg-warning").style.display = s.ffmpeg ? "none" : "block";
    if (s.supported_extensions) {
      supportedExt = new Set(s.supported_extensions);
      fileInput.accept = "audio/*,video/*," + s.supported_extensions.join(",");
    }
    $("#token-state").textContent = s.config.has_hf_token
      ? "✓ A token is saved." : "No token saved yet.";
    if (document.activeElement !== $("#max-jobs")) $("#max-jobs").value = s.config.max_jobs;
    if (document.activeElement !== $("#auth-enabled")) $("#auth-enabled").checked = s.config.auth_enabled;
    if (document.activeElement !== $("#srv-port")) $("#srv-port").value = s.config.port;
    // Don't wipe what's in the box — the user may be mid-edit, or revealed the
    // saved key to read/copy it. Only the placeholder reflects server state.
    if (!$("#api-key").value) {
      $("#api-key").placeholder = s.config.has_api_key
        ? "key is set — click 👁 to reveal" : "set a key…";
    }
    $("#auth-state").textContent = s.config.has_api_key
      ? (s.config.auth_enabled ? "✓ Login is required for other devices." : "A key is saved but login is off — this device and all others have full access.")
      : "No key saved yet — set one before enabling login.";
    return s;
  } catch {
    $("#server-dot").className = "dot err";
    $("#server-label").textContent = "Server unreachable";
    return null;
  }
}

/* ---------------- small helpers ---------------- */
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.classList.add("out"), 4200);
  setTimeout(() => el.remove(), 4700);
}

async function errorText(resp) {
  const err = await resp.json().catch(() => ({}));
  return err.detail || resp.statusText || "request failed";
}

/* POST a JSON body; returns the Response (check .ok) */
function postJSON(url, body) {
  return fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

const store = {
  get(k) { try { return localStorage.getItem("sst." + k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem("sst." + k, v); } catch { /* private mode */ } },
};

function fmtBytes(n) {
  if (!n) return "0 MB";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < 3) { n /= 1024; i++; }
  return n.toFixed(1) + " " + units[i];
}

/* ---------------- step 1: choose files (nothing uploads yet) ---------------- */
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const staged = [];   // [{file, duration}]

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

// filled from /api/status (the server's own list); until then accept anything
// and let the server reject what it can't read
let supportedExt = null;
const extOf = (name) => (name.match(/\.[^.]+$/) || [""])[0].toLowerCase();

function addFiles(list) {
  for (const file of list) {
    if (supportedExt && !supportedExt.has(extOf(file.name))) {
      toast(`"${file.name}" is not an audio/video file this app can read.`, "err");
      continue;
    }
    if (staged.some((s) => s.file.name === file.name && s.file.size === file.size)) continue;
    const item = { file, duration: null, estimate: null };
    staged.push(item);
    probeDuration(item);
  }
  renderStaged();
}

/* Best-effort length read by the browser (works for most mp3/m4a/mp4/wav). */
function probeDuration(item) {
  const url = URL.createObjectURL(item.file);
  const media = document.createElement(item.file.type.startsWith("video") ? "video" : "audio");
  media.preload = "metadata";
  media.onloadedmetadata = () => {
    if (isFinite(media.duration)) { item.duration = media.duration; updateEstimate(); }
    URL.revokeObjectURL(url);
  };
  media.onerror = () => URL.revokeObjectURL(url);
  media.src = url;
}

function renderStaged() {
  const box = $("#staged-list");
  box.innerHTML = "";
  staged.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "staged-row";
    row.innerHTML = `<span class="staged-icon">🎧</span>
      <span class="staged-info"><span class="staged-name"></span>
        <span class="staged-sub">${fmtBytes(item.file.size)}${item.duration ? " · " + fmtDur(item.duration) : ""}${
          item.estimate ? ` · <b>takes about ${fmtDur(item.estimate)}</b>` : ""}</span></span>
      <button class="icon-btn" title="Remove this file" aria-label="Remove">✕</button>`;
    row.querySelector(".staged-name").textContent = item.file.name;
    row.querySelector("button").addEventListener("click", () => { staged.splice(i, 1); updateEstimate(); });
    box.appendChild(row);
  });
  dropzone.classList.toggle("compact", staged.length > 0);
  $("#dropzone .dz-title").textContent = staged.length ? "Add more files" : "Drop audio or video files here";
  const btn = $("#start-btn");
  btn.disabled = !staged.length || uploading;
  btn.textContent = staged.length > 1 ? `Start transcribing ${staged.length} files` : "Start transcribing";
  $("#start-hint").textContent = uploading ? "Uploading…"
    : staged.length ? estimateText() : "Choose a file first.";
}

/* ---------- time estimate before starting ----------
   The server learns its own speed from finished jobs (per model + device), so
   this gets accurate after a job or two. Upload time is not included. */
let queueSeconds = 0;
let estimateIsRough = false;

async function updateEstimate() {
  const diarize = $("#opt-diarize").checked;
  for (const item of staged) {
    if (!item.duration) continue;
    const r = await fetch(`/api/estimate?seconds=${item.duration}&diarize=${diarize}`).catch(() => null);
    if (!r || !r.ok) continue;
    const est = await r.json();
    item.estimate = est.seconds;
    queueSeconds = est.queue_seconds;
    estimateIsRough = !est.learned;
  }
  renderStaged();
}

function estimateText() {
  const known = staged.filter((i) => i.estimate);
  if (!known.length) return "Check the settings, then start.";
  const own = known.reduce((sum, i) => sum + i.estimate, 0);
  let text = `Estimated time: about ${fmtDur(own)}`;
  if (known.length < staged.length) text += " (for the files with a known length)";
  if (queueSeconds > 1) text += `, after ~${fmtDur(queueSeconds)} for jobs already running`;
  if (estimateIsRough) text += " — rough guess until this server has finished a job";
  return text + ".";
}

/* speaker-count lists: "Auto-detect" is in the HTML, 1–10 added here */
for (const sel of [$("#opt-speakers"), $("#rediarize-count")]) {
  for (let n = 1; n <= 10; n++) sel.add(new Option(String(n), String(n)));
}

/* remember the last-used settings on this browser */
for (const id of ["opt-language", "opt-speakers", "opt-script"]) {
  const saved = store.get(id);
  if (saved !== null && [...$("#" + id).options].some((o) => o.value === saved)) $("#" + id).value = saved;
  $("#" + id).addEventListener("change", (e) => store.set(id, e.target.value));
}
if (store.get("opt-diarize") !== null) $("#opt-diarize").checked = store.get("opt-diarize") === "1";
$("#opt-diarize").addEventListener("change", (e) => {
  store.set("opt-diarize", e.target.checked ? "1" : "0");
  $("#opt-speakers").disabled = !e.target.checked;
  updateEstimate();
});
$("#opt-speakers").disabled = !$("#opt-diarize").checked;

/* ---------------- step 3: upload + start ---------------- */
let uploading = false;
let uploadXhr = null;

$("#start-btn").addEventListener("click", startTranscribing);

async function startTranscribing() {
  if (!staged.length || uploading) return;
  uploading = true;
  renderStaged();
  const opts = {
    language: $("#opt-language").value,
    diarize: $("#opt-diarize").checked,
    speakers: $("#opt-speakers").value,
    script: $("#opt-script").value,
  };
  // Upload everything first (the server queues the jobs and runs them one
  // by one), then follow whichever job is running.
  if (watchSource) { watchSource.close(); watchSource = null; watchingJobId = null; }
  while (staged.length) {
    const job = await uploadFile(staged[0].file, opts);
    if (job === "aborted") break;
    staged.shift();
    renderStaged();
    if (job) refreshJobs();
  }
  uploading = false;
  uploadXhr = null;
  renderStaged();
  refreshJobs();
}

function uploadFile(file, opts) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("language", opts.language);
  fd.append("diarize", opts.diarize);
  if (opts.diarize && opts.speakers) fd.append("num_speakers", opts.speakers);
  if (opts.script) fd.append("chinese_script", opts.script);

  showProgressCard(file.name);
  setProgress({ status: "uploading", stage: "uploading", progress: 0 });

  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    uploadXhr = xhr;
    xhr.open("POST", "/api/transcribe");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setProgress({ status: "uploading", stage: "uploading", progress: e.loaded / e.total,
                      detail: `${fmtBytes(e.loaded)} / ${fmtBytes(e.total)}` });
      }
    };
    xhr.onload = () => {
      if (xhr.status === 401) { window.location.href = "/login?next=/"; return; }
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) { resolve(body); return; }
      toast(`Upload of "${file.name}" failed: ${body.detail || xhr.statusText}`, "err");
      hideProgressCard();
      resolve(null);
    };
    xhr.onerror = () => {
      toast(`Upload of "${file.name}" failed — is the server reachable?`, "err");
      hideProgressCard();
      resolve(null);
    };
    xhr.onabort = () => { hideProgressCard(); toast("Upload cancelled."); resolve("aborted"); };
    xhr.send(fd);
  });
}

/* ---------------- progress card ---------------- */
let watchingJobId = null;
let watchSource = null;

function showProgressCard(name) {
  $("#progress-card").classList.remove("hidden");
  $("#progress-file").textContent = name;
  $("#progress-file").title = name;
  setCancelBusy(false);
}

function setCancelBusy(busy) {
  $("#cancel-job").disabled = busy;
  $("#cancel-job").textContent = busy ? "Cancelling…" : "Cancel";
}

function hideProgressCard() {
  $("#progress-card").classList.add("hidden");
}

$("#cancel-job").addEventListener("click", async () => {
  if (uploading && uploadXhr) { uploadXhr.abort(); return; }
  if (!watchingJobId) return;
  await cancelJob(watchingJobId);
});

async function cancelJob(jobId) {
  if (jobId === watchingJobId) {
    setCancelBusy(true);
    $("#progress-stage").textContent = STAGE_LABEL.cancelling;
  }
  const resp = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  if (!resp.ok) {
    toast("Could not cancel: " + await errorText(resp), "err");
    if (jobId === watchingJobId) setCancelBusy(false);
  }
  refreshJobs();
}

/* Follow one job's progress (server-sent events). Reconnects on network
   blips; when the job ends, moves on to the next queued/running job. */
function watchJob(jobId) {
  if (watchSource) watchSource.close();
  watchingJobId = jobId;
  showProgressCard("");
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  watchSource = es;
  es.onmessage = (e) => {
    const job = JSON.parse(e.data);
    if (job.id !== watchingJobId) return;
    $("#progress-file").textContent = job.filename;
    $("#progress-file").title = job.filename;
    setProgress(job);
    if (isActive(job)) return;
    es.close();
    watchSource = null;
    watchingJobId = null;
    hideProgressCard();
    if (job.status === "done") {
      // don't yank away a transcript the user is in the middle of editing
      if (current.editingIdx === null) showResult(job);
      toast(`✓ "${job.filename}" is done.`, "ok");
    } else if (job.status === "error") {
      toast(`"${job.filename}" failed: ${job.error}`, "err");
    } else {
      toast(`"${job.filename}" was cancelled.`);
    }
    refreshJobs();
    refreshStatus();
  };
  es.onerror = () => {
    es.close();
    if (watchSource !== es) return;
    watchSource = null;
    setTimeout(async () => {
      if (watchingJobId !== jobId) return;
      const r = await fetch(`/api/jobs/${jobId}`).catch(() => null);
      if (r && r.status === 404) { watchingJobId = null; hideProgressCard(); return; }  // job deleted
      watchJob(jobId);
    }, 2000);
  };
}

/* Attach the progress card to whatever is running (page reload, other
   browser started it, next file in a batch…). */
function followActiveJob() {
  if (watchingJobId || uploading) return;
  // running first, else the oldest queued (lastJobs is newest first)
  const active = lastJobs.find((j) => j.status === "running") || lastJobs.findLast((j) => j.status === "queued");
  if (active) watchJob(active.id);
}

function fmtDur(s) {
  if (s == null) return "";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  if (h) return `${h}h ${m}m`;
  return m ? `${m}m ${r}s` : `${r}s`;
}

const STAGE_LABEL = {
  uploading: "Uploading", queued: "Waiting in queue", "loading models": "Loading models",
  decoding: "Reading audio", diarizing: "Finding speakers", transcribing: "Transcribing",
  finalizing: "Finishing", cancelling: "Cancelling", done: "Done", error: "Failed", cancelled: "Cancelled",
};

function setProgress(job) {
  const pct = Math.round((job.progress || 0) * 100);
  $("#progress-fill").style.width = pct + "%";
  $("#progress-pct").textContent = pct + "%";
  $("#progress-stage").textContent = STAGE_LABEL[job.stage] || job.stage || "";
  let detail = job.detail || "";
  if (job.stage === "transcribing" && job.chunks_total) detail = `part ${job.chunks_done} of ${job.chunks_total}`;
  if (job.stage === "queued") {
    const ahead = job.queue_position || 0;
    detail = ahead ? `${ahead} job${ahead > 1 ? "s" : ""} ahead` : "starting soon";
  }
  $("#progress-detail").textContent = detail;
  $("#progress-eta").textContent =
    job.eta_seconds != null && job.status === "running" && !job.cancel_requested
      ? "~" + fmtDur(job.eta_seconds) + " left" : "";
  $("#progress-elapsed").textContent =
    job.elapsed_seconds ? "elapsed " + fmtDur(job.elapsed_seconds) : "";
  if (job.cancel_requested) setCancelBusy(true);
}

/* ================= RESULT: karaoke player + editor ================= */
const player = $("#player");

function ts(sec) {
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = (sec % 60).toFixed(1);
  return (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(s).padStart(4, "0");
}

/* Speaker colours: stable hash of the name, overridable per speaker (10-colour
   palette, persisted with the transcript). */
const PALETTE = 10;

function hashColor(name) {
  let h = 5381;
  for (const ch of name) h = ((h * 33) ^ ch.codePointAt(0)) >>> 0;
  return h % PALETTE;
}

function colorIdx(speaker) {
  const custom = (current.result && current.result.speaker_colors) || {};
  return speaker in custom ? custom[speaker] : hashColor(speaker);
}

function speakerClass(speaker) {
  if (!speaker) return "spk-9";  // unknown speaker (no diarization) — neutral grey
  return "spk-" + colorIdx(speaker);
}

/* speaker is null when no diarization ran; show a placeholder, not "null" */
const UNKNOWN_SPEAKER = "—";
function speakerLabel(speaker) { return speaker || UNKNOWN_SPEAKER; }

function isCJK(ch) {
  if (!ch) return false;
  const c = ch.codePointAt(0);
  return (c >= 0x4e00 && c <= 0x9fff) || (c >= 0x3400 && c <= 0x4dbf) ||
         (c >= 0xf900 && c <= 0xfaff) || (c >= 0x3000 && c <= 0x303f) ||
         (c >= 0xff00 && c <= 0xffef);
}

function joinTexts(a, b) {
  a = a.trim(); b = b.trim();
  if (!a) return b;
  if (!b) return a;
  return a + (isCJK(a[a.length - 1]) && isCJK(b[0]) ? "" : " ") + b;
}

function showResult(job) {
  if (!job.result) return;
  commitEdit(false);
  current.jobId = job.id;
  current.filename = job.filename;
  current.result = job.result;
  current.editingIdx = null;
  current.loop = null;
  $("#result-card").classList.remove("hidden");
  $("#rediarize-panel").classList.add("hidden");
  $$(".job-row").forEach((r) => r.classList.toggle("selected", r.dataset.id === job.id));
  $("#result-title").textContent = job.filename;
  const r = job.result;
  renderResultMeta();
  // e.g. diarization was requested but the diarizer couldn't load — the
  // transcript is still here, just without speaker labels.
  const warn = $("#result-warnings");
  const msgs = (r.warnings || []).map((w) => w.message);
  warn.textContent = msgs.length ? "⚠️ " + msgs.join(" ") : "";
  warn.style.display = msgs.length ? "" : "none";
  const src = `/api/jobs/${job.id}/audio`;
  if (job.has_audio === false) { player.style.display = "none"; }
  else { player.style.display = ""; if (!player.src.endsWith(src)) { player.src = src; } }
  $$("#result-card [data-dl]").forEach((btn) => {
    btn.onclick = () => window.open(`/api/jobs/${job.id}/download?format=${btn.dataset.dl}`, "_blank");
  });
  renderSpeakerBar();
  renderSegments();
}

function renderResultMeta() {
  const r = current.result;
  const n = allSpeakers().length;
  $("#result-meta").textContent =
    `${fmtDur(r.duration)} · ${n} speaker${n === 1 ? "" : "s"} · language: ${LANG_NAME[r.language] || r.language || "auto"} · ` +
    `model: ${r.model}` + (r.edited ? " · edited" : "");
}

/* ---------- speaker bar (rename speakers) ---------- */
function allSpeakers() {
  // nulls (undiarized lines) aren't renameable speakers — skip them
  return [...new Set(current.result.segments.map((s) => s.speaker).filter(Boolean))];
}

function renderSpeakerBar() {
  const bar = $("#speaker-bar");
  bar.innerHTML = "";
  const speakers = allSpeakers();
  if (!speakers.length) return;
  const label = document.createElement("span");
  label.className = "bar-label";
  label.textContent = "Speakers (click to rename, recolour or merge):";
  bar.appendChild(label);
  for (const spk of speakers) {
    const chip = document.createElement("span");
    chip.className = "speaker-chip " + speakerClass(spk);
    chip.textContent = spk;
    chip.title = "Rename this speaker or change its colour";
    chip.addEventListener("click", () => openSpeakerPopover(bar, chip, spk));
    bar.appendChild(chip);
  }
}

function openSpeakerPopover(bar, chip, spk) {
  const pop = document.createElement("span");
  pop.className = "spk-popover";
  let chosen = colorIdx(spk);
  const startColor = chosen;

  const input = document.createElement("input");
  input.value = spk;

  const swatches = document.createElement("span");
  swatches.className = "swatches";
  for (let i = 0; i < PALETTE; i++) {
    const sw = document.createElement("button");
    sw.className = `swatch swatch-${i}` + (i === chosen ? " active" : "");
    sw.title = "Colour " + (i + 1);
    sw.addEventListener("click", () => {
      chosen = i;
      swatches.querySelectorAll(".swatch").forEach((b, j) => b.classList.toggle("active", j === i));
    });
    swatches.appendChild(sw);
  }

  const ok = document.createElement("button");
  ok.className = "icon-btn"; ok.textContent = "✓";
  const cancel = document.createElement("button");
  cancel.className = "icon-btn"; cancel.textContent = "✕";

  const commit = async () => {
    const name = input.value.trim() || spk;
    const merging = name !== spk && allSpeakers().includes(name);
    if (merging && !confirm(`"${name}" already exists.\n\nMerge all of ${spk}'s lines into ${name}?`)) return;
    if (name !== spk) {
      current.result.segments.forEach((s) => { if (s.speaker === spk) s.speaker = name; });
    }
    const colors = { ...(current.result.speaker_colors || {}) };
    delete colors[spk];
    // merging keeps the target's colour unless a different swatch was picked
    if (!merging || chosen !== startColor) colors[name] = chosen;
    current.result.speaker_colors = colors;
    await saveResult();
    renderSpeakerBar(); renderSegments();
  };
  ok.addEventListener("click", commit);
  cancel.addEventListener("click", () => renderSpeakerBar());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    if (e.key === "Escape") renderSpeakerBar();
  });

  pop.append(input, swatches, ok, cancel);
  bar.replaceChild(pop, chip);
  input.focus(); input.select();
}

/* ---------- segment list ---------- */
function renderSegments() {
  const box = $("#segments");
  box.innerHTML = "";
  current.result.segments.forEach((seg, idx) => {
    box.appendChild(current.editingIdx === idx ? buildEditorRow(seg, idx) : buildRow(seg, idx));
  });
  $("#add-line-end").onclick = () => insertLine(current.result.segments.length);
}

function buildRow(seg, idx) {
  const div = document.createElement("div");
  div.className = "segment";
  div.dataset.idx = idx;
  div.innerHTML = `
    <span class="seg-time">${ts(seg.start)} – ${ts(seg.end)}</span>
    <span class="speaker-chip ${speakerClass(seg.speaker)}">${escapeHtml(speakerLabel(seg.speaker))}</span>
    <span class="seg-text"></span>
    <button class="seg-insert" title="Insert a new line below">＋</button>`;
  div.querySelector(".seg-text").textContent = seg.text;
  div.addEventListener("click", (e) => {
    if (e.target.closest(".seg-insert")) return;
    if (player.src) { player.currentTime = seg.start + 0.01; player.play(); }
  });
  div.addEventListener("dblclick", (e) => {
    if (e.target.closest(".seg-insert")) return;
    enterEdit(idx);
  });
  div.querySelector(".seg-insert").addEventListener("click", () => insertLine(idx + 1));
  return div;
}

/* ---------- karaoke highlight ---------- */
player.addEventListener("timeupdate", () => {
  if (!current.result) return;
  const t = player.currentTime;
  if (current.loop && t >= current.loop.end - 0.05) {
    player.currentTime = current.loop.start + 0.01;
    return;
  }
  if (current.editingIdx !== null) return;
  let active = -1;
  current.result.segments.forEach((seg, i) => { if (t >= seg.start && t < seg.end) active = i; });
  $$("#segments .segment").forEach((el) => {
    const on = Number(el.dataset.idx) === active;
    if (on && !el.classList.contains("playing") && !player.paused) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    el.classList.toggle("playing", on);
  });
});

/* ---------- editing ---------- */
function enterEdit(idx) {
  commitEdit(false);
  current.editingIdx = idx;
  renderSegments();
  const seg = current.result.segments[idx];
  if (player.src && seg.end > seg.start) {
    current.loop = { start: seg.start, end: seg.end };
    player.currentTime = seg.start + 0.01;
    player.play().catch(() => {});
  }
  const ta = $("#segments textarea");
  if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
}

function exitEdit() {
  // Cancelling the edit of a line with no text (e.g. a freshly added line)
  // removes it instead of leaving an empty line behind.
  const idx = current.editingIdx;
  if (idx !== null && current.result.segments[idx] && !current.result.segments[idx].text.trim()) {
    current.result.segments.splice(idx, 1);
  }
  current.editingIdx = null;
  current.loop = null;
  renderSegments();
}

function commitEdit(rerender = true) {
  if (current.editingIdx === null) return;
  const ta = $("#segments textarea");
  const sel = $("#segments .seg-toolbar select");
  const idx = current.editingIdx;
  const seg = current.result.segments[idx];
  // stale word timings are dropped by the server when the text changed
  if (ta) seg.text = ta.value.trim();
  if (sel && sel.value !== "__new__") seg.speaker = sel.value || null;  // "" = unassigned
  if (!seg.text) current.result.segments.splice(idx, 1);  // saving an empty line deletes it
  current.editingIdx = null;
  current.loop = null;
  saveResult();
  if (rerender) renderSegments();
  renderSpeakerBar();
}

function buildEditorRow(seg, idx) {
  const div = document.createElement("div");
  div.className = "segment editing";
  div.dataset.idx = idx;
  const editor = document.createElement("div");
  editor.className = "seg-editor";

  const toolbar = document.createElement("div");
  toolbar.className = "seg-toolbar";
  toolbar.innerHTML = `
    <span class="seg-time">${ts(seg.start)} – ${ts(seg.end)}</span>
    <select title="Speaker for this line"></select>
    <button class="icon-btn" data-act="done" title="Save line (Enter)">✓ Done</button>
    <button class="icon-btn" data-act="split" title="Split into two lines at the cursor (Shift+Enter)">✂ Split</button>
    <button class="icon-btn" data-act="merge" title="Merge with the previous line (Backspace at line start)">⇧ Merge up</button>
    <button class="icon-btn" data-act="insert" title="Insert a new empty line below">＋ Line below</button>
    <button class="icon-btn danger" data-act="delete" title="Delete this line">✕ Delete</button>
    <button class="icon-btn" data-act="cancel" title="Discard changes (Esc)">Cancel</button>`;

  const select = toolbar.querySelector("select");
  // Lines with no speaker (diarization off/unavailable) start unassigned and
  // can be given one here.
  const noneOpt = document.createElement("option");
  noneOpt.value = ""; noneOpt.textContent = UNKNOWN_SPEAKER + " (no speaker)";
  if (!seg.speaker) noneOpt.selected = true;
  select.appendChild(noneOpt);
  for (const spk of allSpeakers()) {
    const opt = document.createElement("option");
    opt.value = spk; opt.textContent = spk;
    if (spk === seg.speaker) opt.selected = true;
    select.appendChild(opt);
  }
  const newOpt = document.createElement("option");
  newOpt.value = "__new__"; newOpt.textContent = "＋ New speaker…";
  select.appendChild(newOpt);
  select.addEventListener("change", () => {
    if (select.value === "__new__") {
      const name = prompt("New speaker name:", "SPEAKER_" + String(allSpeakers().length).padStart(2, "0"));
      if (name && name.trim()) {
        const opt = document.createElement("option");
        opt.value = name.trim(); opt.textContent = name.trim();
        select.insertBefore(opt, newOpt);
        select.value = name.trim();
      } else {
        select.value = seg.speaker || "";
      }
    }
  });

  const ta = document.createElement("textarea");
  ta.value = seg.text;
  ta.rows = Math.max(1, Math.ceil(seg.text.length / 60));
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); splitLine(idx, ta); }
    else if (e.key === "Enter") { e.preventDefault(); commitEdit(); }
    else if (e.key === "Escape") { e.preventDefault(); exitEdit(); }
    else if (e.key === "Backspace" && ta.selectionStart === 0 && ta.selectionEnd === 0 && idx > 0) {
      e.preventDefault(); mergeUp(idx, ta.value);
    }
  });

  toolbar.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (!act) return;
    if (act === "done") commitEdit();
    else if (act === "cancel") exitEdit();
    else if (act === "split") splitLine(idx, ta);
    else if (act === "merge") mergeUp(idx, ta.value);
    else if (act === "delete") deleteLine(idx);
    else if (act === "insert") { commitEdit(false); insertLine(idx + 1); }
  });

  editor.appendChild(toolbar);
  editor.appendChild(ta);
  div.appendChild(editor);
  return div;
}

function splitLine(idx, ta) {
  const seg = current.result.segments[idx];
  const pos = ta.selectionStart;
  const left = ta.value.slice(0, pos).trim();
  const right = ta.value.slice(pos).trim();
  if (!left || !right) return;
  const frac = Math.min(0.95, Math.max(0.05, pos / ta.value.length));
  const mid = seg.start + (seg.end - seg.start) * frac;
  const rightSeg = { start: Math.round(mid * 1000) / 1000, end: seg.end, speaker: seg.speaker, text: right };
  seg.text = left;
  seg.end = rightSeg.start;
  current.result.segments.splice(idx + 1, 0, rightSeg);
  current.editingIdx = null;
  current.loop = null;
  saveResult();
  renderSegments();
}

function mergeUp(idx, currentText) {
  if (idx <= 0) return;
  const prev = current.result.segments[idx - 1];
  const seg = current.result.segments[idx];
  if (prev.words && seg.words) prev.words = prev.words.concat(seg.words);  // server drops them if they no longer match
  prev.text = joinTexts(prev.text, currentText);
  prev.end = Math.max(prev.end, seg.end);
  current.result.segments.splice(idx, 1);
  current.editingIdx = null;
  current.loop = null;
  saveResult();
  renderSegments();
  renderSpeakerBar();
}

function deleteLine(idx) {
  current.result.segments.splice(idx, 1);
  current.editingIdx = null;
  current.loop = null;
  saveResult();
  renderSegments();
  renderSpeakerBar();
}

function insertLine(idx) {
  const segs = current.result.segments;
  const prev = segs[idx - 1], next = segs[idx];
  const start = prev ? prev.end : (next ? Math.max(0, next.start - 2) : 0);
  const end = next ? Math.min(next.start, start + 2) : start + 2;
  const speaker = prev ? prev.speaker : (next ? next.speaker : "SPEAKER_00");
  segs.splice(idx, 0, { start, end: Math.max(end, start + 0.2), speaker, text: "" });
  current.editingIdx = null;
  enterEdit(idx);
}

async function saveResult() {
  if (!current.jobId || !current.result) return;
  const resp = await fetch(`/api/jobs/${current.jobId}/result`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segments: current.result.segments,
      speaker_colors: current.result.speaker_colors || {},
    }),
  });
  if (resp.ok) {
    const { speakers, text, edited, speaker_colors } = await resp.json();
    Object.assign(current.result, { speakers, text, edited, speaker_colors });
    renderResultMeta();
  } else {
    toast("Could not save edit: " + await errorText(resp), "err");
  }
}

/* ---------- convert Simplified <-> Traditional ---------- */
$("#convert-script").addEventListener("change", async (e) => {
  const target = e.target.value;
  e.target.value = "";
  if (!target || !current.jobId) return;
  commitEdit(false);
  const resp = await postJSON(`/api/jobs/${current.jobId}/convert`, { script: target });
  if (!resp.ok) { toast("Could not convert: " + await errorText(resp), "err"); return; }
  const { result } = await resp.json();
  showResult({ ...lastJobs.find((j) => j.id === current.jobId), id: current.jobId,
               filename: current.filename, result });
  toast("Converted. Exports now use the converted text.", "ok");
});

/* ---------- re-detect speakers ---------- */
$("#rediarize-open").addEventListener("click", () => {
  if (!current.result) return;
  const panel = $("#rediarize-panel");
  panel.classList.toggle("hidden");
  if (!panel.classList.contains("hidden")) {
    const n = allSpeakers().length;
    $("#rediarize-count").value = n >= 1 && n <= 10 ? String(n) : "";
    $("#rediarize-count").focus();
  }
});
$("#rediarize-cancel").addEventListener("click", () => $("#rediarize-panel").classList.add("hidden"));
$("#rediarize-go").addEventListener("click", async () => {
  if (!current.jobId) return;
  commitEdit(false);
  const btn = $("#rediarize-go");
  btn.disabled = true;
  const count = $("#rediarize-count").value;
  const resp = await postJSON(`/api/jobs/${current.jobId}/rediarize`, { num_speakers: count ? Number(count) : null });
  btn.disabled = false;
  if (!resp.ok) { toast("Could not start: " + await errorText(resp), "err"); return; }
  $("#rediarize-panel").classList.add("hidden");
  toast("Re-detecting speakers — the new transcript will open when it is ready.");
  refreshJobs();
});

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------- job history ---------------- */
const LANG_NAME = { yue: "Cantonese", zh: "Mandarin", en: "English" };

function fmtWhen(epoch) {
  if (!epoch) return "";
  const d = new Date(epoch * 1000);
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " +
      d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function jobMeta(job) {
  const bits = [];
  if (job.audio_duration) bits.push(fmtDur(job.audio_duration));
  if (job.status === "done") {
    if (job.speaker_count) bits.push(`${job.speaker_count} speaker${job.speaker_count > 1 ? "s" : ""}`);
    const lang = job.language || job.params.language;
    if (lang) bits.push(LANG_NAME[lang] || lang);
  } else if (job.params.num_speakers) {
    bits.push(`${job.params.num_speakers} speakers`);
  }
  if (job.params.rediarize_from) bits.push("re-detected");
  bits.push(fmtWhen(job.created_at));
  return bits.filter(Boolean).join(" · ");
}

async function refreshJobs() {
  const resp = await fetch("/api/jobs").catch(() => null);
  if (!resp || !resp.ok) return;
  lastJobs = (await resp.json()).jobs;
  followActiveJob();
  const list = $("#job-list");
  list.innerHTML = "";
  if (!lastJobs.length) { list.innerHTML = '<div class="empty">No jobs yet — finished transcripts appear here.</div>'; return; }
  for (const job of lastJobs) {
    const active = isActive(job);
    const row = document.createElement("div");
    row.className = "job-row" + (job.id === current.jobId ? " selected" : "");
    row.dataset.id = job.id;
    row.tabIndex = 0;
    const statusText = job.stage === "cancelling" ? "cancelling" : job.status;
    row.innerHTML = `
      <div class="job-main">
        <div class="job-name"></div>
        <div class="job-meta"><span class="job-status ${job.status}">${statusText}</span>
          <span class="job-sub"></span></div>
        ${job.status === "running" ? `<div class="job-bar"><div style="width:${Math.round(job.progress * 100)}%"></div></div>` : ""}
      </div>
      <button class="icon-btn danger job-act"></button>`;
    row.querySelector(".job-name").textContent = job.filename;
    row.querySelector(".job-name").title = job.filename;
    row.querySelector(".job-sub").textContent = jobMeta(job);
    const btn = row.querySelector(".job-act");
    if (active) {
      btn.textContent = "■";
      btn.title = "Cancel this job";
      btn.setAttribute("aria-label", "Cancel " + job.filename);
      btn.disabled = job.cancel_requested;
      btn.addEventListener("click", (e) => { e.stopPropagation(); cancelJob(job.id); });
    } else {
      btn.textContent = "🗑";
      btn.title = "Delete this job (audio + transcript)";
      btn.setAttribute("aria-label", "Delete " + job.filename);
      btn.addEventListener("click", (e) => { e.stopPropagation(); deleteJob(job); });
    }
    const open = () => openJob(job);
    row.addEventListener("click", open);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    list.appendChild(row);
  }
}

async function openJob(job) {
  if (isActive(job)) { watchJob(job.id); return; }
  if (job.status === "error") { toast("This job failed: " + job.error, "err"); return; }
  if (job.status === "cancelled") { toast("This job was cancelled — there is no transcript."); return; }
  const resp = await fetch(`/api/jobs/${job.id}`);
  if (!resp.ok) { toast("Could not open: " + await errorText(resp), "err"); return; }
  showResult(await resp.json());
  $("#result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function deleteJob(job) {
  if (!confirm(`Delete "${job.filename}"?\n\nThe transcript and the stored audio are removed for good.`)) return;
  const resp = await fetch(`/api/jobs/${job.id}`, { method: "DELETE" });
  if (!resp.ok) { toast("Could not delete: " + await errorText(resp), "err"); }
  else if (current.jobId === job.id) {
    player.pause();
    player.removeAttribute("src");
    $("#result-card").classList.add("hidden");
    current.jobId = null;
    current.result = null;
  }
  refreshJobs();
}

/* ---------------- models ---------------- */
let modelsPollTimer = null;

async function refreshModels() {
  const data = await fetch("/api/models").then((r) => r.json()).catch(() => null);
  if (!data) return;
  renderCatalog($("#stt-catalog"), data.stt);
  renderCatalog($("#diar-catalog"), data.diarization);
  renderSelectors(data);
  syncSearchRows(data.stt);

  const busy = [...data.stt, ...data.diarization].some(
    (m) => m.download && m.download.status === "downloading");
  clearTimeout(modelsPollTimer);
  if (busy) modelsPollTimer = setTimeout(refreshModels, 1200);
}

/* Keep Hugging Face search rows in sync with download progress. */
function syncSearchRows(sttEntries) {
  $$("#hf-results [data-search-repo]").forEach((cell) => {
    const m = sttEntries.find((e) => e.repo_id === cell.dataset.searchRepo);
    if (!m) return;
    if (m.download && m.download.status === "downloading") {
      cell.innerHTML = `<div class="dl-progress">${dlProgressText(m.download)}</div>
                        <button class="btn danger small" data-cancel-dl="${m.repo_id}">✕ Cancel</button>`;
      cell.querySelector("[data-cancel-dl]").addEventListener("click", () => cancelDownload(m.repo_id));
    } else if (m.download && m.download.status === "cancelled" && !m.downloaded) {
      cell.innerHTML = `<button class="btn primary" data-dl-repo="${m.repo_id}">Download</button>`;
      cell.querySelector("[data-dl-repo]").addEventListener("click", async (e) => {
        e.target.disabled = true; e.target.textContent = "Starting…";
        await postJSON("/api/models/download", { repo_id: m.repo_id });
        refreshModels();
      });
    } else if (m.download && m.download.status === "error") {
      cell.innerHTML = `<div class="dl-error">${escapeHtml(m.download.error || "")}</div>
                        <button class="btn" data-dl-repo="${m.repo_id}">Retry</button>`;
      cell.querySelector("[data-dl-repo]").addEventListener("click", async (e) => {
        e.target.disabled = true;
        await postJSON("/api/models/download", { repo_id: m.repo_id });
        refreshModels();
      });
    } else if (m.downloaded) {
      cell.innerHTML = `<span class="badge ok">downloaded</span>
                        <button class="btn danger small" data-rm-repo="${m.repo_id}">Remove</button>`;
      cell.querySelector("[data-rm-repo]").addEventListener("click", () => removeModel(m.repo_id).then(refreshModels));
    }
  });
}

function renderSelectors(data) {
  const fill = (sel, entries) => {
    const el = $(sel);
    const prev = el.value;
    el.innerHTML = "";
    const available = entries.filter((e) => e.downloaded && e.engine !== "unknown");
    if (!available.length) {
      el.innerHTML = '<option value="">— download a model first —</option>';
      return;
    }
    for (const e of available) {
      const opt = document.createElement("option");
      opt.value = e.repo_id;
      opt.textContent = e.display_name + (e.loaded ? "  ✓ loaded" : "");
      if (e.selected || e.repo_id === prev) opt.selected = true;
      el.appendChild(opt);
    }
  };
  fill("#sel-stt", data.stt);
  fill("#sel-diar", data.diarization);
}

function dlProgressText(dl) {
  if (!dl.total_bytes) return "downloading… " + fmtBytes(dl.downloaded_bytes);
  const pct = Math.round(dl.progress * 100);
  let eta = "";
  if (dl.eta_seconds != null && dl.eta_seconds > 1) {
    eta = dl.eta_seconds > 86400 ? " · slow connection" : ` · ~${fmtDur(dl.eta_seconds)} left`;
  }
  return `${pct}% · ${fmtBytes(dl.downloaded_bytes)} / ${fmtBytes(dl.total_bytes)}${eta}`;
}

function renderCatalog(container, entries) {
  container.innerHTML = "";
  for (const m of entries) {
    const row = document.createElement("div");
    row.className = "model-row";
    const badges = [
      m.loaded ? '<span class="badge loaded">loaded</span>' : "",
      m.downloaded ? '<span class="badge ok">downloaded</span>' : "",
      m.gated ? '<span class="badge gated">needs HF token</span>' : "",
    ].join("");
    let action;
    if (m.download && m.download.status === "downloading") {
      action = `<div class="dl-progress">${dlProgressText(m.download)}</div>
                <button class="btn danger small" data-cancel-dl="${m.repo_id}">✕ Cancel</button>`;
    } else if (m.download && m.download.status === "error") {
      action = `<div class="dl-error">${escapeHtml(m.download.error || "")}</div>
                <button class="btn" data-dl-repo="${m.repo_id}">Retry</button>`;
    } else if (m.downloaded) {
      action = (m.loaded || m.selected)
        ? `<span class="hint" style="margin:0">in use</span>`
        : `<button class="btn danger small" data-rm-repo="${m.repo_id}">Remove</button>`;
    } else {
      action = `<button class="btn primary" data-dl-repo="${m.repo_id}">Download ${m.size}</button>`;
    }
    const licenseCls = /AMBIGUOUS/.test(m.license || "") ? "warn" : "ok";
    row.innerHTML = `
      <div class="model-info">
        <div class="model-name">${m.display_name} ${badges}</div>
        <div class="model-desc"><strong>${m.languages}</strong> · ${m.size}<br>${m.strengths}
        ${m.requires_extra ? `<br><em>Requires: uv sync --extra ${m.requires_extra}</em>` : ""}</div>
        ${m.license ? `<div class="model-license ${licenseCls}">⚖ ${m.license}</div>` : ""}
        ${m.downloaded && m.attribution ? `<div class="model-license">© ${m.attribution}</div>` : ""}
      </div>
      <div class="model-dl">${action}</div>`;
    container.appendChild(row);
  }
  container.querySelectorAll("[data-dl-repo]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      const resp = await postJSON("/api/models/download", { repo_id: btn.dataset.dlRepo });
      if (!resp.ok) toast(await errorText(resp), "err");
      refreshModels();
    });
  });
  container.querySelectorAll("[data-rm-repo]").forEach((btn) => {
    btn.addEventListener("click", () => removeModel(btn.dataset.rmRepo));
  });
  container.querySelectorAll("[data-cancel-dl]").forEach((btn) => {
    btn.addEventListener("click", () => cancelDownload(btn.dataset.cancelDl));
  });
}

async function cancelDownload(repoId) {
  if (!confirm(`Cancel downloading "${repoId}"? Partially downloaded files will be removed.`)) return;
  await postJSON("/api/models/download/cancel", { repo_id: repoId });
  refreshModels();
}

async function removeModel(repoId) {
  if (!confirm(`Remove "${repoId}" from local storage? You can re-download it later.`)) return;
  const resp = await postJSON("/api/models/remove", { repo_id: repoId });
  if (!resp.ok) toast("Could not remove: " + await errorText(resp), "err");
  refreshModels();
}

$("#apply-models").addEventListener("click", async () => {
  const stt = $("#sel-stt").value, diar = $("#sel-diar").value;
  if (!stt) { toast("Download a speech-to-text model first.", "err"); return; }
  const body = { stt_model: stt, load_now: true };
  if (diar) body.diarization_model = diar;
  await postJSON("/api/config", body);
  $("#apply-models").textContent = "Loading…";
  setTimeout(() => { $("#apply-models").textContent = "Apply & load"; refreshModels(); refreshStatus(); }, 3000);
});

/* ---------------- HF search ---------------- */
async function hfSearch() {
  const q = $("#hf-query").value.trim();
  if (!q) return;
  const box = $("#hf-results");
  box.innerHTML = '<div class="empty">Searching…</div>';
  const resp = await fetch(`/api/models/search?q=${encodeURIComponent(q)}`);
  if (!resp.ok) { box.innerHTML = '<div class="empty">Search failed — are you online?</div>'; return; }
  const data = await resp.json();
  box.innerHTML = data.results.length ? "" : '<div class="empty">No results.</div>';
  for (const m of data.results) {
    const row = document.createElement("div");
    row.className = "model-row";
    const compat = m.compatible
      ? '<span class="badge ok">compatible</span>'
      : '<span class="badge gated" title="This repo is in a format for another runtime (MLX, GGUF, CTranslate2, ONNX…) or an unsupported architecture. Look for the standard PyTorch version of the same model.">unsupported format</span>';
    row.innerHTML = `
      <div class="model-info">
        <div class="model-name">${m.repo_id} ${compat} ${m.gated ? '<span class="badge gated">gated</span>' : ""}</div>
        <div class="model-desc">${(m.downloads || 0).toLocaleString()} downloads · ${(m.likes || 0)} likes ·
          check the model page for its license before commercial use</div>
      </div>
      <div class="model-dl" data-search-repo="${m.repo_id}">
        ${m.downloaded ? `<button class="btn danger small" data-rm-repo="${m.repo_id}">Remove</button>`
          : m.compatible ? `<button class="btn primary" data-dl-repo="${m.repo_id}">Download</button>` : ""}
      </div>`;
    const btn = row.querySelector("[data-dl-repo]");
    if (btn) btn.addEventListener("click", async () => {
      btn.disabled = true; btn.textContent = "Starting…";
      await postJSON("/api/models/download", { repo_id: m.repo_id });
      refreshModels();  // the models poll now drives live progress in this row
    });
    const rm = row.querySelector("[data-rm-repo]");
    if (rm) rm.addEventListener("click", async () => { await removeModel(m.repo_id); hfSearch(); });
    box.appendChild(row);
  }
}
$("#hf-search-btn").addEventListener("click", hfSearch);
$("#hf-query").addEventListener("keydown", (e) => { if (e.key === "Enter") hfSearch(); });

/* ---------------- settings ---------------- */
$("#save-token").addEventListener("click", async () => {
  await postJSON("/api/config", { hf_token: $("#hf-token").value.trim(), load_now: false });
  $("#hf-token").value = "";
  $("#token-state").textContent = "✓ Token saved.";
  refreshStatus();
});
$("#save-device").addEventListener("click", async () => {
  await postJSON("/api/config", { device_override: $("#sel-device").value, load_now: false });
  toast("Saved. Device changes apply the next time models load (restart the server to force).", "ok");
});
$("#save-max-jobs").addEventListener("click", async () => {
  const v = parseInt($("#max-jobs").value, 10);
  const resp = await postJSON("/api/config", { max_jobs: v, load_now: false });
  if (!resp.ok) toast(await errorText(resp), "err");
  else { $("#save-max-jobs").textContent = "Saved ✓"; setTimeout(() => $("#save-max-jobs").textContent = "Save", 1500); }
  refreshJobs();
});

/* ---------------- access & security (API key + auth toggle) ---------------- */
$("#key-reveal").addEventListener("click", async () => {
  const input = $("#api-key");
  if (input.type === "text") { input.type = "password"; return; }
  // Nothing typed yet: pull the saved key so it can actually be read/copied.
  if (!input.value) {
    const resp = await fetch("/api/config/api-key");
    if (resp.ok) input.value = (await resp.json()).api_key || "";
  }
  input.type = "text";
});
$("#key-regen").addEventListener("click", async () => {
  const resp = await fetch("/api/config/regenerate-key", { method: "POST" });
  if (!resp.ok) { toast("Could not generate a key.", "err"); return; }
  const data = await resp.json();
  const input = $("#api-key");
  input.type = "text";
  input.value = data.api_key;
  $("#auth-state").textContent = "New key generated and saved. Copy it and update any clients using the old key.";
});
$("#key-copy").addEventListener("click", async () => {
  const input = $("#api-key");
  if (!input.value) { toast("Nothing to copy — reveal or regenerate a key first.", "err"); return; }
  await navigator.clipboard.writeText(input.value);
});
$("#save-key").addEventListener("click", async () => {
  const value = $("#api-key").value;
  if (!value) { refreshStatus(); return; }  // blank = keep existing key
  const resp = await postJSON("/api/config", { api_key: value, load_now: false });
  if (!resp.ok) {
    toast(await errorText(resp), "err");
    return;
  }
  refreshStatus();
});
$("#auth-enabled").addEventListener("change", async (e) => {
  const want = e.target.checked;
  const resp = await postJSON("/api/config", { auth_enabled: want, load_now: false });
  if (!resp.ok) {
    toast(await errorText(resp), "err");
    e.target.checked = !want;
    return;
  }
  refreshStatus();
});

/* ---------------- server port ---------------- */
$("#save-port").addEventListener("click", async () => {
  const p = parseInt($("#srv-port").value, 10);
  const resp = await postJSON("/api/config", { port: p, load_now: false });
  if (!resp.ok) {
    toast(await errorText(resp), "err");
    return;
  }
  const data = await resp.json();
  if (data.restart) {
    $("#save-port").textContent = "Restarting…";
    setTimeout(() => {
      const url = new URL(window.location.href);
      url.port = String(data.port);
      window.location.href = url.toString();
    }, 1200);
  }
});

/* ---------------- dashboard: share access ---------------- */
$$("[data-copy-target]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const el = document.getElementById(btn.dataset.copyTarget);
    if (el) await navigator.clipboard.writeText(el.textContent);
  });
});

async function refreshNetwork() {
  const data = await fetch("/api/network").then((r) => r.json()).catch(() => null);
  if (!data) return;
  $("#share-local").textContent = `http://localhost:${data.port}`;
  $("#share-api").textContent = data.addresses.length
    ? `http://${data.addresses[0]}:${data.port}/v1`
    : `http://localhost:${data.port}/v1`;

  const rows = $("#share-lan-rows");
  rows.innerHTML = "";
  const addRow = (label, url) => {
    const div = document.createElement("div");
    div.className = "share-row";
    const id = "share-lan-" + Math.random().toString(36).slice(2);
    div.innerHTML = `<span class="share-label">${label}</span><code id="${id}"></code>
      <button class="icon-btn" data-copy-target="${id}" title="Copy">⧉</button>`;
    div.querySelector("code").textContent = url;
    rows.appendChild(div);
    div.querySelector("[data-copy-target]").addEventListener("click", async () => {
      await navigator.clipboard.writeText(url);
    });
  };
  for (const ip of data.addresses) addRow("WiFi / Ethernet", `http://${ip}:${data.port}`);
  if (data.mdns) addRow("mDNS (Apple/most OSes)", `http://${data.mdns}:${data.port}`);
  for (const ip of (data.vpn_addresses || [])) addRow("Tailscale / VPN", `http://${ip}:${data.port}`);
  if (!data.addresses.length && !(data.vpn_addresses || []).length)
    rows.innerHTML = '<div class="hint" style="margin:0">No network address detected — is this machine on WiFi/Ethernet?</div>';

  if (data.auth_enabled) {
    $("#share-auth-state").textContent = "🔒 Login required — share the API key separately.";
    $("#share-curl").style.display = "";
    const ip = data.addresses[0] || "localhost";
    $("#share-curl-code").textContent =
      `curl -H "Authorization: Bearer <key>" http://${ip}:${data.port}/v1/models`;
  } else {
    $("#share-auth-state").textContent =
      "🔓 Open on the LAN — anyone who can reach this address can use it. Enable login in Settings to restrict.";
    $("#share-curl").style.display = "none";
  }
}

/* ---------------- dashboard: server log ---------------- */
let lastLogCount = -1;

async function refreshLogs() {
  const resp = await fetch("/api/logs?lines=300").catch(() => null);
  if (!resp || !resp.ok) return;
  const data = await resp.json();
  if (document.activeElement !== $("#log-mode")) $("#log-mode").value = data.access_log;
  if (data.count === lastLogCount) return;  // nothing new
  lastLogCount = data.count;
  const view = $("#log-view");
  const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 20;
  view.textContent = data.lines.length ? data.lines.join("\n") : "(empty)";
  if (atBottom) view.scrollTop = view.scrollHeight;
  $("#log-file-hint").textContent =
    `Full log file on the server: ${data.file} (rotates at 5 MB, keeps 3 old files).`;
}
$("#log-refresh").addEventListener("click", refreshLogs);
$("#log-download").addEventListener("click", () => window.open("/api/logs/download", "_blank"));
$("#log-clear").addEventListener("click", async () => {
  if (!confirm("Clear the log? The current log file is moved to sst.log.1.")) return;
  const resp = await fetch("/api/logs/clear", { method: "POST" });
  if (!resp.ok) toast("Could not clear: " + await errorText(resp), "err");
  refreshLogs();
});
$("#log-mode").addEventListener("change", async (e) => {
  const resp = await postJSON("/api/config", { access_log: e.target.value, load_now: false });
  if (!resp.ok) toast(await errorText(resp), "err");
  else toast("Log setting saved.", "ok");
});

/* ---------------- init + polling ----------------
   Poll only while this tab is visible: a backgrounded browser tab
   shouldn't keep hitting the server (and filling its log). */
function activeTab() { return $(".nav-item.active").dataset.tab; }

function poll() {
  if (document.hidden) return;
  refreshStatus();
  refreshJobs();
  if (activeTab() === "dashboard") refreshLogs();
}

refreshStatus();
refreshJobs();
refreshModels();
refreshNetwork();
setInterval(poll, 10000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
