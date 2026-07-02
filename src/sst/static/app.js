/* SST web UI — transcribe, karaoke playback, live transcript editing */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

/* Current transcript being shown/edited */
const current = { jobId: null, filename: "", result: null, editingIdx: null, loop: null };
let speakerColors = {};
let watchingJobId = null;

/* ---------------- tabs ---------------- */
$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".nav-item").forEach((b) => b.classList.remove("active"));
    $$(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "models") refreshModels();
    if (btn.dataset.tab === "dashboard") refreshStatus();
  });
});

/* ---------------- status ---------------- */
async function refreshStatus() {
  try {
    const s = await fetch("/api/status").then((r) => r.json());
    $("#server-dot").className = "dot ok";
    $("#server-label").textContent = "Server running";
    $("#device-label").textContent = s.device_description;
    $("#d-server").textContent = "Running · v" + s.version;
    $("#d-device").textContent = s.device_description;
    $("#d-stt").textContent = s.stt_loaded || "not loaded yet";
    $("#d-diar").textContent = s.diarization_loaded || "not loaded yet";
    $("#dash-attribution").style.display =
      s.diarization_loaded === "pyannote/speaker-diarization-community-1" ? "block" : "none";
    $("#ffmpeg-warning").style.display = s.ffmpeg ? "none" : "block";
    $("#token-state").textContent = s.config.has_hf_token
      ? "✓ A token is saved." : "No token saved yet.";
    if (document.activeElement !== $("#max-jobs")) $("#max-jobs").value = s.config.max_jobs;
    return s;
  } catch {
    $("#server-dot").className = "dot err";
    $("#server-label").textContent = "Server unreachable";
    return null;
  }
}

/* ---------------- upload ---------------- */
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) submitFile(fileInput.files[0]);
  fileInput.value = "";
});
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) submitFile(e.dataTransfer.files[0]);
});

async function submitFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("language", $("#opt-language").value);
  fd.append("diarize", $("#opt-diarize").checked);
  const spk = $("#opt-speakers").value;
  if (spk) fd.append("num_speakers", spk);

  $("#result-card").classList.add("hidden");
  $("#progress-card").classList.remove("hidden");
  $("#progress-file").textContent = file.name;
  setProgress({ stage: "uploading", progress: 0, elapsed_seconds: 0 });

  const resp = await fetch("/api/transcribe", { method: "POST", body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    alert("Upload failed: " + (err.detail || resp.statusText));
    $("#progress-card").classList.add("hidden");
    return;
  }
  const job = await resp.json();
  watchingJobId = job.id;
  watchJob(job.id);
  refreshJobs();
}

$("#cancel-job").addEventListener("click", async () => {
  if (!watchingJobId) return;
  await fetch(`/api/jobs/${watchingJobId}/cancel`, { method: "POST" });
});

function watchJob(jobId) {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.onmessage = (e) => {
    const job = JSON.parse(e.data);
    if (job.id === watchingJobId) setProgress(job);
    if (job.status === "done") {
      es.close();
      if (job.id === watchingJobId) {
        $("#progress-card").classList.add("hidden");
        showResult(job);
      }
      refreshJobs(); refreshStatus();
    } else if (job.status === "error" || job.status === "cancelled") {
      es.close();
      if (job.id === watchingJobId) {
        $("#progress-card").classList.add("hidden");
        if (job.status === "error") alert("Transcription failed: " + job.error);
      }
      refreshJobs();
    }
  };
  es.onerror = () => es.close();
}

function fmtDur(s) {
  if (s == null) return "";
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return m ? `${m}m ${r}s` : `${r}s`;
}

function setProgress(job) {
  const pct = Math.round((job.progress || 0) * 100);
  $("#progress-fill").style.width = pct + "%";
  $("#progress-pct").textContent = pct + "%";
  $("#progress-stage").textContent = job.stage || "";
  $("#progress-eta").textContent =
    job.eta_seconds != null && job.status === "running" && job.stage === "transcribing"
      ? "~" + fmtDur(job.eta_seconds) + " remaining" : "";
  $("#progress-elapsed").textContent =
    job.elapsed_seconds ? "elapsed " + fmtDur(job.elapsed_seconds) : "";
}

/* ================= RESULT: karaoke player + editor ================= */
const player = $("#player");

function ts(sec) {
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = (sec % 60).toFixed(1);
  return (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(s).padStart(4, "0");
}

function speakerClass(speaker) {
  if (!(speaker in speakerColors)) speakerColors[speaker] = Object.keys(speakerColors).length % 6;
  return "spk-" + speakerColors[speaker];
}

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
  speakerColors = {};
  $("#result-card").classList.remove("hidden");
  $("#result-title").textContent = job.filename;
  const r = job.result;
  $("#result-meta").textContent =
    `${fmtDur(r.duration)} · ${(r.speakers || []).length} speaker(s) · language: ${r.language || "auto"} · ` +
    `model: ${r.model}` + (r.edited ? " · edited" : "");
  const src = `/api/jobs/${job.id}/audio`;
  if (job.has_audio === false) { player.style.display = "none"; }
  else { player.style.display = ""; if (!player.src.endsWith(src)) { player.src = src; } }
  $$("#result-card [data-dl]").forEach((btn) => {
    btn.onclick = () => window.open(`/api/jobs/${job.id}/download?format=${btn.dataset.dl}`, "_blank");
  });
  renderSpeakerBar();
  renderSegments();
}

/* ---------- speaker bar (rename speakers) ---------- */
function allSpeakers() {
  return [...new Set(current.result.segments.map((s) => s.speaker))];
}

function renderSpeakerBar() {
  const bar = $("#speaker-bar");
  bar.innerHTML = "";
  const speakers = allSpeakers();
  if (!speakers.length) return;
  const label = document.createElement("span");
  label.className = "bar-label";
  label.textContent = "Speakers (click to rename):";
  bar.appendChild(label);
  for (const spk of speakers) {
    const chip = document.createElement("span");
    chip.className = "speaker-chip " + speakerClass(spk);
    chip.textContent = spk;
    chip.title = "Rename this speaker everywhere";
    chip.addEventListener("click", () => {
      const input = document.createElement("input");
      input.value = spk;
      bar.replaceChild(input, chip);
      input.focus(); input.select();
      const commit = async () => {
        const name = input.value.trim();
        if (name && name !== spk) {
          current.result.segments.forEach((s) => { if (s.speaker === spk) s.speaker = name; });
          await saveResult();
        }
        renderSpeakerBar(); renderSegments();
      };
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); commit(); }
        if (e.key === "Escape") { renderSpeakerBar(); }
      });
      input.addEventListener("blur", commit);
    });
    bar.appendChild(chip);
  }
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
    <span class="speaker-chip ${speakerClass(seg.speaker)}">${escapeHtml(seg.speaker)}</span>
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
  if (ta) seg.text = ta.value.trim();
  if (sel && sel.value !== "__new__") seg.speaker = sel.value;
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
        select.value = seg.speaker;
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
    body: JSON.stringify({ segments: current.result.segments }),
  });
  if (resp.ok) {
    const data = await resp.json();
    current.result = data.result;
    $("#result-meta").textContent = $("#result-meta").textContent.includes("· edited")
      ? $("#result-meta").textContent
      : $("#result-meta").textContent + " · edited";
  } else {
    alert("Could not save edit: " + ((await resp.json().catch(() => ({}))).detail || resp.statusText));
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------- job history ---------------- */
async function refreshJobs() {
  const data = await fetch("/api/jobs").then((r) => r.json()).catch(() => null);
  if (!data) return;
  const list = $("#job-list");
  list.innerHTML = "";
  if (!data.jobs.length) { list.innerHTML = '<div class="empty">No jobs yet</div>'; return; }
  for (const job of data.jobs) {
    const row = document.createElement("div");
    row.className = "job-row";
    row.innerHTML = `
      <span class="job-name"></span>
      <span class="job-sub">${job.audio_duration ? fmtDur(job.audio_duration) : ""}</span>
      <span class="job-status ${job.status}">${job.status}</span>
      <span class="job-actions"></span>`;
    row.querySelector(".job-name").textContent = job.filename;
    const actions = row.querySelector(".job-actions");
    if (job.status === "running" || job.status === "queued") {
      const btn = document.createElement("button");
      btn.className = "icon-btn danger"; btn.textContent = "■ Cancel"; btn.title = "Cancel this job";
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" });
        refreshJobs();
      });
      actions.appendChild(btn);
    } else {
      const btn = document.createElement("button");
      btn.className = "icon-btn danger"; btn.textContent = "🗑"; btn.title = "Delete this job (audio + transcript)";
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete "${job.filename}" (audio + transcript)?`)) return;
        await fetch(`/api/jobs/${job.id}`, { method: "DELETE" });
        if (current.jobId === job.id) { $("#result-card").classList.add("hidden"); current.jobId = null; }
        refreshJobs();
      });
      actions.appendChild(btn);
    }
    row.addEventListener("click", async () => {
      const full = await fetch(`/api/jobs/${job.id}`).then((r) => r.json());
      if (full.status === "done") {
        showResult(full);
        $("#result-card").scrollIntoView({ behavior: "smooth" });
      } else if (full.status === "running" || full.status === "queued") {
        watchingJobId = full.id;
        $("#progress-card").classList.remove("hidden");
        $("#progress-file").textContent = full.filename;
        watchJob(full.id);
      } else if (full.status === "error") {
        alert("This job failed: " + full.error);
      }
    });
    list.appendChild(row);
  }
}

/* ---------------- models ---------------- */
let modelsPollTimer = null;

async function refreshModels() {
  const data = await fetch("/api/models").then((r) => r.json()).catch(() => null);
  if (!data) return;
  renderCatalog($("#stt-catalog"), data.stt);
  renderCatalog($("#diar-catalog"), data.diarization);
  renderSelectors(data);

  const busy = [...data.stt, ...data.diarization].some(
    (m) => m.download && m.download.status === "downloading");
  clearTimeout(modelsPollTimer);
  if (busy) modelsPollTimer = setTimeout(refreshModels, 1200);
}

function renderSelectors(data) {
  const fill = (sel, entries) => {
    const el = $(sel);
    const prev = el.value;
    el.innerHTML = "";
    const available = entries.filter((e) => e.downloaded);
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

function fmtBytes(n) {
  if (!n) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < 3) { n /= 1024; i++; }
  return n.toFixed(1) + " " + units[i];
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
      const pct = Math.round(m.download.progress * 100);
      action = `<div class="dl-progress">${pct}% · ${fmtBytes(m.download.downloaded_bytes)} / ${fmtBytes(m.download.total_bytes)}</div>`;
    } else if (m.download && m.download.status === "error") {
      action = `<div class="dl-error">${m.download.error}</div>
                <button class="btn" data-dl-repo="${m.repo_id}">Retry</button>`;
    } else if (m.downloaded) {
      action = "";
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
      const resp = await fetch("/api/models/download", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: btn.dataset.dlRepo }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        alert(err.detail || "Download failed to start");
      }
      refreshModels();
    });
  });
}

$("#apply-models").addEventListener("click", async () => {
  const stt = $("#sel-stt").value, diar = $("#sel-diar").value;
  if (!stt) { alert("Download an STT model first."); return; }
  const body = { stt_model: stt, load_now: true };
  if (diar) body.diarization_model = diar;
  await fetch("/api/config", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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
      : '<span class="badge gated">unsupported architecture</span>';
    row.innerHTML = `
      <div class="model-info">
        <div class="model-name">${m.repo_id} ${compat} ${m.gated ? '<span class="badge gated">gated</span>' : ""}</div>
        <div class="model-desc">${(m.downloads || 0).toLocaleString()} downloads · ${(m.likes || 0)} likes ·
          check the model page for its license before commercial use</div>
      </div>
      <div class="model-dl">
        ${m.downloaded ? '<span class="badge ok">downloaded</span>'
          : m.compatible ? `<button class="btn primary" data-dl-repo="${m.repo_id}">Download</button>` : ""}
      </div>`;
    const btn = row.querySelector("[data-dl-repo]");
    if (btn) btn.addEventListener("click", async () => {
      btn.disabled = true; btn.textContent = "Downloading…";
      await fetch("/api/models/download", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: m.repo_id }),
      });
    });
    box.appendChild(row);
  }
}
$("#hf-search-btn").addEventListener("click", hfSearch);
$("#hf-query").addEventListener("keydown", (e) => { if (e.key === "Enter") hfSearch(); });

/* ---------------- settings ---------------- */
$("#save-token").addEventListener("click", async () => {
  await fetch("/api/config", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hf_token: $("#hf-token").value.trim(), load_now: false }),
  });
  $("#hf-token").value = "";
  $("#token-state").textContent = "✓ Token saved.";
  refreshStatus();
});
$("#save-device").addEventListener("click", async () => {
  await fetch("/api/config", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_override: $("#sel-device").value, load_now: false }),
  });
  alert("Saved. Device changes apply the next time models load (restart the server to force).");
});
$("#save-max-jobs").addEventListener("click", async () => {
  const v = parseInt($("#max-jobs").value, 10);
  const resp = await fetch("/api/config", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_jobs: v, load_now: false }),
  });
  if (!resp.ok) alert("Value must be between 3 and 20.");
  else { $("#save-max-jobs").textContent = "Saved ✓"; setTimeout(() => $("#save-max-jobs").textContent = "Save", 1500); }
  refreshJobs();
});

/* ---------------- init ---------------- */
refreshStatus();
refreshJobs();
refreshModels();
setInterval(refreshStatus, 10000);
setInterval(refreshJobs, 8000);
