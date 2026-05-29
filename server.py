import tempfile
import threading
import uuid
import os
import json
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import av
from fastapi import Depends, FastAPI, File, Form, HTTPException, Header, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

app = FastAPI(title="Local Speech-to-Text API")

API_KEY = os.getenv("STT_API_KEY", "").strip()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "STT_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:9000,http://127.0.0.1:9000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
    cpu_threads=2,
    num_workers=1,
)

jobs = {}
jobs_lock = threading.Lock()
history_lock = threading.Lock()
UPLOAD_CHUNK_SIZE = 1024 * 1024
HISTORY_DIR = Path("transcripts_history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _safe_base_name(filename: str | None) -> str:
    raw = Path(filename or "transcript").stem
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw)
    return safe.strip("-_") or "transcript"


def _persist_history(filename: str | None, text_output: str, json_output: dict) -> dict:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    history_id = f"{ts}-{uuid.uuid4().hex[:8]}"
    record = {
        "id": history_id,
        "filename": filename or "unknown",
        "basename": _safe_base_name(filename),
        "created_at": now.isoformat(),
        "result_text": text_output,
        "result_json": json_output,
    }
    target = HISTORY_DIR / f"{history_id}.json"
    with history_lock:
        with target.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
    return {
        "id": record["id"],
        "filename": record["filename"],
        "basename": record["basename"],
        "created_at": record["created_at"],
    }


def _list_history(page: int = 1, page_size: int = 20) -> dict:
    items = []
    with history_lock:
        files = sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.name, reverse=True)
    total_items = len(files)
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 1
    if page_size > 100:
        page_size = 100
    start = (page - 1) * page_size
    end = start + page_size
    for path in files[start:end]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items.append(
                {
                    "id": data.get("id", path.stem),
                    "filename": data.get("filename", "unknown"),
                    "basename": data.get("basename", _safe_base_name(data.get("filename"))),
                    "created_at": data.get("created_at"),
                }
            )
        except Exception:
            continue
    total_pages = (total_items + page_size - 1) // page_size if total_items else 1
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
    }


def _read_history(history_id: str) -> dict:
    path = HISTORY_DIR / f"{history_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="History entry not found")
    with history_lock:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _audio_duration_seconds(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                return max(0.0, container.duration / 1_000_000)
            for stream in container.streams:
                if stream.type == "audio" and stream.duration is not None and stream.time_base is not None:
                    return max(0.0, float(stream.duration * stream.time_base))
    except Exception:
        return 0.0
    return 0.0


def _run_job(job_id: str, temp_path: Path, original_name: str) -> None:
    try:
        total_seconds = _audio_duration_seconds(temp_path)

        with jobs_lock:
            if jobs[job_id]["status"] == "cancel_requested":
                jobs[job_id]["status"] = "canceled"
                jobs[job_id]["message"] = "Transcription canceled"
                return
            jobs[job_id]["status"] = "transcribing"
            jobs[job_id]["message"] = "Transcribing audio"
            jobs[job_id]["total_seconds"] = total_seconds

        segments, info = model.transcribe(
            str(temp_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        segment_list = []
        lines = []

        for segment in segments:
            with jobs_lock:
                if jobs[job_id]["status"] == "cancel_requested":
                    jobs[job_id]["status"] = "canceled"
                    jobs[job_id]["message"] = "Transcription canceled"
                    return

            text = segment.text.strip()
            segment_list.append({"start": segment.start, "end": segment.end, "text": text})
            lines.append(f"[{segment.start:.2f} - {segment.end:.2f}] {text}")

            if total_seconds > 0:
                pct = min(99.0, max(0.0, (segment.end / total_seconds) * 100))
            else:
                pct = min(99.0, len(segment_list) * 2.0)

            with jobs_lock:
                jobs[job_id]["progress"] = pct
                jobs[job_id]["processed_seconds"] = segment.end

        text_output = "\n".join(lines)
        json_output = {
            "filename": original_name,
            "language": info.language,
            "segments": segment_list,
            "text": " ".join(item["text"] for item in segment_list),
        }
        history_entry = _persist_history(original_name, text_output, json_output)

        with jobs_lock:
            if jobs[job_id]["status"] == "cancel_requested":
                jobs[job_id]["status"] = "canceled"
                jobs[job_id]["message"] = "Transcription canceled"
                return
            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = "Transcription complete"
            jobs[job_id]["progress"] = 100.0
            jobs[job_id]["result_text"] = text_output
            jobs[job_id]["result_json"] = json_output
            jobs[job_id]["history_id"] = history_entry["id"]
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = str(exc)
    finally:
        temp_path.unlink(missing_ok=True)


def _cancel_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in {"done", "error", "canceled"}:
            return dict(job)
        job["status"] = "cancel_requested"
        job["message"] = "Cancel requested"
        return dict(job)


async def _save_upload_to_temp(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio").suffix or ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp_path = Path(temp.name)
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            temp.write(chunk)

    return temp_path


async def _create_job(file: UploadFile, response_format: str) -> dict:
    temp_path = await _save_upload_to_temp(file)

    job_id = uuid.uuid4().hex

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "filename": file.filename,
            "response_format": response_format,
            "processed_seconds": 0.0,
            "total_seconds": 0.0,
            "result_text": None,
            "result_json": None,
        }

    thread = threading.Thread(target=_run_job, args=(job_id, temp_path, file.filename), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


def _format_transcript_response(text_output: str, json_output: dict, response_format: str):
    if response_format == "json":
        return JSONResponse(json_output)
    return PlainTextResponse(text_output)


def _vtt_timestamp_to_seconds(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0
    return 0.0


def _parse_vtt(vtt_text: str) -> list[dict]:
    segments = []
    lines = vtt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue

        start_raw, end_raw = [part.strip().split(" ")[0] for part in line.split("-->", 1)]
        start = _vtt_timestamp_to_seconds(start_raw)
        end = _vtt_timestamp_to_seconds(end_raw)
        i += 1

        text_lines = []
        while i < len(lines) and lines[i].strip() != "":
            text = re.sub(r"<[^>]+>", "", lines[i].strip())
            text = html.unescape(text)
            if text:
                text_lines.append(text)
            i += 1

        text = " ".join(text_lines).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
        i += 1

    return segments


def _select_caption_track(info: dict) -> tuple[str, dict] | tuple[None, None]:
    tracks_by_lang = {}
    for source_key in ("subtitles", "automatic_captions"):
        source = info.get(source_key) or {}
        if isinstance(source, dict):
            tracks_by_lang.update(source)

    if not tracks_by_lang:
        return None, None

    language_order = ["en", "en-US", "en-us", "en-x-autogen"]
    language_order.extend(lang for lang in tracks_by_lang if str(lang).lower().startswith("en"))

    seen = set()
    for language in language_order:
        if language in seen or language not in tracks_by_lang:
            continue
        seen.add(language)
        tracks = tracks_by_lang.get(language) or []
        for track in tracks:
            protocol = (track.get("protocol") or "").lower() if isinstance(track, dict) else ""
            url = str(track.get("url") or "") if isinstance(track, dict) else ""
            if (
                isinstance(track, dict)
                and url
                and (track.get("ext") or "").lower() == "vtt"
                and protocol != "m3u8_native"
                and ".vtt" in url
            ):
                return language, track
        for track in tracks:
            if isinstance(track, dict) and track.get("url"):
                return language, track

    return None, None


def _fetch_url_transcript(url: str) -> tuple[str, dict]:
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

        language, track = _select_caption_track(info)
        if not track:
            raise HTTPException(status_code=404, detail="No Vimeo subtitles/transcript track found for this URL")

        response = ydl.urlopen(str(track["url"]))
        vtt_text = response.read().decode("utf-8", errors="replace")

    segments = _parse_vtt(vtt_text)
    if not segments:
        raise HTTPException(status_code=422, detail="Subtitle track was found but could not be parsed")

    title = str(info.get("title") or info.get("id") or "vimeo-transcript")
    text_output = "\n".join(f"[{item['start']:.2f} - {item['end']:.2f}] {item['text']}" for item in segments)
    json_output = {
        "filename": title,
        "source_url": url,
        "source": "vimeo_captions",
        "language": language,
        "segments": segments,
        "text": " ".join(item["text"] for item in segments),
    }
    _persist_history(title, text_output, json_output)
    return text_output, json_output


@app.get("/")
def health_check():
    return {"status": "ok", "model": "base.en", "device": "cpu"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Local Speech-to-Text</title>
  <style>
    :root {
      --bg0: #f5f7ff;
      --bg1: #e8fbf5;
      --card: #ffffff;
      --ink: #13253f;
      --muted: #5b6f88;
      --accent: #0f8f76;
      --accent-2: #2d6cdf;
      --border: #dbe5f1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(1200px 700px at 5% -10%, #d7e7ff 0%, transparent 60%),
        radial-gradient(900px 600px at 95% 0%, #d7fff1 0%, transparent 50%),
        linear-gradient(140deg, var(--bg0), var(--bg1));
      padding: 24px;
    }
    .wrap { max-width: 860px; margin: 0 auto; }
    .card {
      position: relative;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 12px 34px rgba(11, 33, 63, 0.1);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .title-wrap { min-width: 0; }
    h1 { margin: 0 0 8px; letter-spacing: 0.2px; }
    .muted { color: var(--muted); margin: 0 0 18px; }
    .row { margin: 14px 0; }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr auto auto;
      align-items: center;
    }
    input[type=file] {
      width: 100%;
      border: 1px solid var(--border);
      background: #fafdff;
      border-radius: 10px;
      color: var(--ink);
      padding: 10px 12px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 16px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }
    .cancel-button { background: #d94848; }
    .reset-button {
      background: #facc15;
      color: #1f2937;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
    }
    .reset-icon {
      font-size: 14px;
      line-height: 1;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .statusline { margin: 10px 0 0; color: var(--muted); font-size: 14px; min-height: 20px; }
    .progress-wrap { margin-top: 14px; }
    .hidden { display: none; }
    .progress-label { font-size: 13px; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
    .download-actions { display: flex; gap: 10px; margin-top: 14px; }
    .tabs { display: flex; gap: 10px; margin-top: 16px; }
    .tab-btn { background: #e2e8f0; color: #0f172a; }
    .tab-btn.active { background: #2563eb; color: #fff; }
    .history-actions { margin-top: 12px; }
    .history-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .history-page-info { color: var(--muted); font-size: 13px; }
    .history-list {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      max-height: 280px;
      overflow: auto;
      background: #f9fbff;
    }
    .history-item {
      width: 100%;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--border);
      border-radius: 0;
      background: transparent;
      color: var(--ink);
      padding: 10px 12px;
      font-weight: 500;
    }
    .history-item:last-child { border-bottom: 0; }
    .history-meta { color: var(--muted); font-size: 12px; }
    .progress {
      width: 100%;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: #e8eef8;
      border: 1px solid var(--border);
    }
    .bar {
      width: 0%;
      height: 100%;
      transition: width .25s linear;
      background: linear-gradient(90deg, #14a58a, #2d6cdf);
    }
    .busy {
      animation: pulse 1s ease-in-out infinite alternate;
    }
    @keyframes pulse {
      from { opacity: .6; }
      to { opacity: 1; }
    }
    pre {
      margin-top: 16px;
      background: #f6f9ff;
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 10px;
      white-space: pre-wrap;
      max-height: 420px;
      overflow: auto;
    }
    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
      .download-actions { flex-direction: column; }
      button { width: 100%; }
      .reset-button { width: auto; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <div class=\"card-header\">
        <div class=\"title-wrap\">
          <h1>Local Speech-to-Text</h1>
          <p class=\"muted\">Upload audio/video and track progress while transcribing.</p>
        </div>
        <button id=\"reset-btn\" class=\"reset-button\" type=\"button\" title=\"Reset\">
          <span class=\"reset-icon\" aria-hidden=\"true\">↺</span>
          <span>Reset</span>
        </button>
      </div>

      <form id=\"upload-form\">
        <div class=\"row grid\">
          <input id=\"file\" name=\"file\" type=\"file\" accept=\"audio/*,video/*\" required />

          <button id=\"submit-btn\" type=\"submit\">Transcribe</button>
          <button id=\"cancel-btn\" class=\"cancel-button\" type=\"button\" disabled>Cancel</button>
        </div>
      </form>

      <p id=\"status\" class=\"statusline\"></p>

      <div id=\"upload-progress-wrap\" class=\"progress-wrap hidden\">
        <div class=\"progress-label\"><span>Upload</span><span id=\"upload-pct\">0%</span></div>
        <div class=\"progress\"><div id=\"upload-bar\" class=\"bar\"></div></div>
      </div>

      <div id=\"transcribe-progress-wrap\" class=\"progress-wrap hidden\">
        <div class=\"progress-label\"><span>Transcription</span><span id=\"transcribe-pct\">0%</span></div>
        <div class=\"progress\"><div id=\"transcribe-bar\" class=\"bar\"></div></div>
      </div>

      <div id=\"download-actions\" class=\"download-actions hidden\">
        <button id=\"download-text-btn\" type=\"button\" disabled>Download TXT</button>
        <button id=\"download-json-btn\" type=\"button\" disabled>Download JSON</button>
      </div>

      <div class=\"tabs\">
        <button id=\"result-tab-btn\" class=\"tab-btn active\" type=\"button\">Result</button>
        <button id=\"history-tab-btn\" class=\"tab-btn\" type=\"button\">History</button>
      </div>

      <div id=\"result-panel\">
        <pre id=\"result\"></pre>
      </div>

      <div id=\"history-panel\" class=\"hidden\">
        <div class=\"history-actions\">
          <div class=\"history-controls\">
            <button id=\"refresh-history-btn\" type=\"button\">Refresh History</button>
            <button id=\"history-prev-btn\" type=\"button\">Previous</button>
            <button id=\"history-next-btn\" type=\"button\">Next</button>
            <span id=\"history-page-info\" class=\"history-page-info\"></span>
          </div>
        </div>
        <div id=\"history-list\" class=\"history-list\"></div>
      </div>
    </div>
  </div>

  <script>
    const form = document.getElementById('upload-form');
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    const submitBtn = document.getElementById('submit-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const resetBtn = document.getElementById('reset-btn');
    const downloadTextBtn = document.getElementById('download-text-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const resultTabBtn = document.getElementById('result-tab-btn');
    const historyTabBtn = document.getElementById('history-tab-btn');
    const resultPanel = document.getElementById('result-panel');
    const historyPanel = document.getElementById('history-panel');
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    const historyPrevBtn = document.getElementById('history-prev-btn');
    const historyNextBtn = document.getElementById('history-next-btn');
    const historyPageInfo = document.getElementById('history-page-info');
    const historyList = document.getElementById('history-list');
    const uploadProgressWrap = document.getElementById('upload-progress-wrap');
    const transcribeProgressWrap = document.getElementById('transcribe-progress-wrap');
    const downloadActions = document.getElementById('download-actions');
    const uploadBar = document.getElementById('upload-bar');
    const uploadPct = document.getElementById('upload-pct');
    const transcribeBar = document.getElementById('transcribe-bar');
    const transcribePct = document.getElementById('transcribe-pct');

    let pollTimer = null;
    let activeXhr = null;
    let activeJobId = null;
    let lastResultText = '';
    let lastResultJson = null;
    let lastResultBaseName = 'transcript';
    let historyEntries = [];
    let historyPage = 1;
    let historyPageSize = 20;
    let historyTotalPages = 1;
    let historyTotalItems = 0;

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function setActiveTab(tab) {
      const resultActive = tab === 'result';
      resultTabBtn.classList.toggle('active', resultActive);
      historyTabBtn.classList.toggle('active', !resultActive);
      resultPanel.classList.toggle('hidden', !resultActive);
      historyPanel.classList.toggle('hidden', resultActive);
    }

    function hydrateResult(entry) {
      lastResultText = entry.result_text || '';
      lastResultJson = entry.result_json || null;
      lastResultBaseName = safeBaseName(entry.filename || 'transcript');
      result.textContent = lastResultText;
      downloadTextBtn.disabled = !lastResultText;
      downloadJsonBtn.disabled = !lastResultJson;
      downloadActions.classList.remove('hidden');
      status.textContent = `Loaded history: ${entry.filename || entry.id}`;
      setActiveTab('result');
    }

    async function loadHistoryDetail(id) {
      const res = await fetch(`/ui/history/${id}`);
      if (!res.ok) {
        throw new Error(`History fetch failed (${res.status})`);
      }
      const entry = await res.json();
      hydrateResult(entry);
    }

    function renderHistory() {
      if (!historyEntries.length) {
        historyList.innerHTML = '<div class="history-item">No history yet.</div>';
        historyPageInfo.textContent = `Page ${historyPage} of ${historyTotalPages} (${historyTotalItems} total)`;
        historyPrevBtn.disabled = historyPage <= 1;
        historyNextBtn.disabled = historyPage >= historyTotalPages;
        return;
      }
      const dtf = new Intl.DateTimeFormat(undefined, {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
      historyList.innerHTML = historyEntries.map((entry) => {
        const filename = escapeHtml(entry.filename || 'unknown');
        const createdRaw = entry.created_at || '';
        const createdFormatted = createdRaw ? dtf.format(new Date(createdRaw)) : '';
        const created = escapeHtml(createdFormatted);
        const id = escapeHtml(entry.id);
        return `<button class="history-item" type="button" data-history-id="${id}"><div>${filename}</div><div class="history-meta">${created}</div></button>`;
      }).join('');
      historyPageInfo.textContent = `Page ${historyPage} of ${historyTotalPages} (${historyTotalItems} total)`;
      historyPrevBtn.disabled = historyPage <= 1;
      historyNextBtn.disabled = historyPage >= historyTotalPages;
    }

    async function refreshHistory(page = historyPage) {
      const targetPage = Math.max(1, page);
      const res = await fetch(`/ui/history?page=${targetPage}&page_size=${historyPageSize}`);
      if (!res.ok) {
        throw new Error(`History list failed (${res.status})`);
      }
      const data = await res.json();
      historyEntries = Array.isArray(data.items) ? data.items : [];
      historyPage = Number(data.page) || 1;
      historyPageSize = Number(data.page_size) || historyPageSize;
      historyTotalPages = Number(data.total_pages) || 1;
      historyTotalItems = Number(data.total_items) || 0;
      renderHistory();
    }

    function setUploadProgress(pct) {
      const clamped = Math.max(0, Math.min(100, pct));
      uploadBar.style.width = clamped + '%';
      uploadPct.textContent = clamped.toFixed(0) + '%';
    }

    function setTranscriptionProgress(pct) {
      const clamped = Math.max(0, Math.min(100, pct));
      transcribeBar.style.width = clamped + '%';
      transcribePct.textContent = clamped.toFixed(0) + '%';
    }

    function resetProgress() {
      setUploadProgress(0);
      setTranscriptionProgress(0);
      transcribeBar.classList.remove('busy');
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      activeXhr = null;
      activeJobId = null;
      lastResultText = '';
      lastResultJson = null;
      lastResultBaseName = 'transcript';
      cancelBtn.disabled = true;
      downloadTextBtn.disabled = true;
      downloadJsonBtn.disabled = true;
      uploadProgressWrap.classList.add('hidden');
      transcribeProgressWrap.classList.add('hidden');
      downloadActions.classList.add('hidden');
    }

    function safeBaseName(filename) {
      const rawName = filename || 'transcript';
      const withoutExt = rawName.replace(/[.][^/.]+$/, '');
      return withoutExt.replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '') || 'transcript';
    }

    function downloadBlob(filename, content, type) {
      const blob = new Blob([content], { type });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    async function pollJob(jobId, format) {
      if (pollTimer) {
        clearInterval(pollTimer);
      }

      const tick = async () => {
        try {
          const res = await fetch(`/ui/jobs/${jobId}`);
          if (!res.ok) {
            throw new Error(`Status ${res.status}`);
          }

          const data = await res.json();
          if (typeof data.progress === 'number') {
            setTranscriptionProgress(data.progress);
          }

          if (data.status === 'transcribing') {
            transcribeProgressWrap.classList.remove('hidden');
            transcribeBar.classList.add('busy');
            status.textContent = data.message || 'Transcribing audio...';
            return;
          }

          if (data.status === 'queued') {
            status.textContent = data.message || 'Queued...';
            return;
          }

          if (data.status === 'done') {
            clearInterval(pollTimer);
            pollTimer = null;
            transcribeBar.classList.remove('busy');
            setTranscriptionProgress(100);
            status.textContent = 'Done.';
            activeJobId = null;
            lastResultText = data.result_text || '';
            lastResultJson = data.result_json || null;
            lastResultBaseName = safeBaseName(data.filename || (data.result_json && data.result_json.filename));
            result.textContent = lastResultText;
            submitBtn.disabled = false;
            cancelBtn.disabled = true;
            downloadTextBtn.disabled = !lastResultText;
            downloadJsonBtn.disabled = !lastResultJson;
            downloadActions.classList.remove('hidden');
            refreshHistory().catch(() => {});
            return;
          }

          if (data.status === 'cancel_requested' || data.status === 'canceled') {
            clearInterval(pollTimer);
            pollTimer = null;
            transcribeBar.classList.remove('busy');
            status.textContent = 'Canceled.';
            result.textContent = '';
            activeJobId = null;
            submitBtn.disabled = false;
            cancelBtn.disabled = true;
            return;
          }

          if (data.status === 'error') {
            clearInterval(pollTimer);
            pollTimer = null;
            transcribeBar.classList.remove('busy');
            status.textContent = 'Transcription failed.';
            result.textContent = data.message || 'Unknown error';
            activeJobId = null;
            submitBtn.disabled = false;
            cancelBtn.disabled = true;
            return;
          }
        } catch (error) {
          clearInterval(pollTimer);
          pollTimer = null;
          transcribeBar.classList.remove('busy');
          status.textContent = 'Progress check failed.';
          result.textContent = String(error);
          activeJobId = null;
          submitBtn.disabled = false;
          cancelBtn.disabled = true;
        }
      };

      await tick();
      pollTimer = setInterval(tick, 1500);
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      result.textContent = '';
      resetProgress();
      submitBtn.disabled = true;
      cancelBtn.disabled = false;

      const fileInput = document.getElementById('file');
      const format = 'text';

      if (!fileInput.files.length) {
        status.textContent = 'Choose a file first.';
        submitBtn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('response_format', format);

      status.textContent = 'Uploading file...';
      uploadProgressWrap.classList.remove('hidden');

      const xhr = new XMLHttpRequest();
      activeXhr = xhr;
      xhr.open('POST', '/ui/transcribe-jobs');

      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) {
          setUploadProgress((ev.loaded / ev.total) * 100);
        }
      };

      xhr.upload.onload = () => {
        setUploadProgress(100);
        status.textContent = 'Upload complete. Starting transcription...';
        transcribeProgressWrap.classList.remove('hidden');
        transcribeBar.classList.add('busy');
      };

      xhr.onerror = () => {
        status.textContent = 'Upload failed.';
        result.textContent = 'Network error while uploading file.';
        activeXhr = null;
        submitBtn.disabled = false;
        cancelBtn.disabled = true;
      };

      xhr.onabort = () => {
        status.textContent = 'Upload canceled.';
        result.textContent = '';
        activeXhr = null;
        submitBtn.disabled = false;
        cancelBtn.disabled = true;
      };

      xhr.onload = async () => {
        activeXhr = null;
        if (xhr.status < 200 || xhr.status >= 300) {
          status.textContent = `Request failed: ${xhr.status}`;
          result.textContent = xhr.responseText;
          submitBtn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }

        let payload;
        try {
          payload = JSON.parse(xhr.responseText);
        } catch (error) {
          status.textContent = 'Could not parse server response.';
          result.textContent = String(error);
          submitBtn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }

        if (!payload.job_id) {
          status.textContent = 'Server did not return a job id.';
          result.textContent = xhr.responseText;
          submitBtn.disabled = false;
          cancelBtn.disabled = true;
          return;
        }

        activeJobId = payload.job_id;
        status.textContent = 'Transcribing audio...';
        await pollJob(payload.job_id, format);
      };

      xhr.send(formData);
    });

    function resetForm() {
      if (activeXhr && activeXhr.readyState !== XMLHttpRequest.DONE) {
        activeXhr.abort();
      }
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      if (activeJobId) {
        fetch(`/ui/jobs/${activeJobId}/cancel`, { method: 'POST' }).catch(() => {});
      }
      form.reset();
      result.textContent = '';
      status.textContent = '';
      transcribeBar.classList.remove('busy');
      resetProgress();
      submitBtn.disabled = false;
    }

    cancelBtn.addEventListener('click', async () => {
      cancelBtn.disabled = true;

      if (activeXhr && activeXhr.readyState !== XMLHttpRequest.DONE) {
        activeXhr.abort();
        return;
      }

      if (!activeJobId) {
        submitBtn.disabled = false;
        status.textContent = 'Canceled.';
        return;
      }

      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }

      try {
        await fetch(`/ui/jobs/${activeJobId}/cancel`, { method: 'POST' });
      } catch (error) {
        result.textContent = String(error);
      }

      activeJobId = null;
      transcribeBar.classList.remove('busy');
      status.textContent = 'Cancel requested.';
      submitBtn.disabled = false;
    });

    resetBtn.addEventListener('click', resetForm);
    resultTabBtn.addEventListener('click', () => setActiveTab('result'));
    historyTabBtn.addEventListener('click', () => {
      setActiveTab('history');
      refreshHistory().catch((error) => {
        status.textContent = 'Failed to load history.';
        result.textContent = String(error);
      });
    });
    refreshHistoryBtn.addEventListener('click', () => {
      refreshHistory(1).catch((error) => {
        status.textContent = 'Failed to refresh history.';
        result.textContent = String(error);
      });
    });
    historyPrevBtn.addEventListener('click', () => {
      if (historyPage <= 1) {
        return;
      }
      refreshHistory(historyPage - 1).catch((error) => {
        status.textContent = 'Failed to load previous history page.';
        result.textContent = String(error);
      });
    });
    historyNextBtn.addEventListener('click', () => {
      if (historyPage >= historyTotalPages) {
        return;
      }
      refreshHistory(historyPage + 1).catch((error) => {
        status.textContent = 'Failed to load next history page.';
        result.textContent = String(error);
      });
    });
    historyList.addEventListener('click', (event) => {
      const target = event.target.closest('[data-history-id]');
      if (!target) {
        return;
      }
      const id = target.getAttribute('data-history-id');
      if (!id) {
        return;
      }
      loadHistoryDetail(id).catch((error) => {
        status.textContent = 'Failed to load history entry.';
        result.textContent = String(error);
      });
    });

    downloadTextBtn.addEventListener('click', () => {
      if (!lastResultText) {
        return;
      }
      downloadBlob(`${lastResultBaseName}.txt`, lastResultText, 'text/plain;charset=utf-8');
    });

    downloadJsonBtn.addEventListener('click', () => {
      if (!lastResultJson) {
        return;
      }
      downloadBlob(
        `${lastResultBaseName}.json`,
        JSON.stringify(lastResultJson, null, 2),
        'application/json;charset=utf-8'
      );
    });
    refreshHistory().catch(() => {});
  </script>
</body>
</html>
"""


@app.post("/transcribe-jobs")
async def create_transcription_job(
    file: UploadFile = File(...),
    response_format: str = Form("text"),
    _: None = Depends(require_api_key),
):
    return await _create_job(file, response_format)

@app.post("/ui/transcribe-jobs")
async def create_ui_transcription_job(
    file: UploadFile = File(...),
    response_format: str = Form("text"),
):
    return await _create_job(file, response_format)


@app.post("/transcribe-url")
def transcribe_url(
    url: str = Form(...),
    response_format: str = Form("json"),
    _: None = Depends(require_api_key),
):
    text_output, json_output = _fetch_url_transcript(url)
    return _format_transcript_response(text_output, json_output, response_format)


@app.post("/ui/transcribe-url")
def transcribe_ui_url(
    url: str = Form(...),
    response_format: str = Form("json"),
):
    text_output, json_output = _fetch_url_transcript(url)
    return _format_transcript_response(text_output, json_output, response_format)


@app.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    _: None = Depends(require_api_key),
):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    _: None = Depends(require_api_key),
):
    return _cancel_job(job_id)


@app.get("/ui/jobs/{job_id}")
def get_ui_job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.post("/ui/jobs/{job_id}/cancel")
def cancel_ui_job(job_id: str):
    return _cancel_job(job_id)


@app.get("/ui/history")
def list_ui_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    return _list_history(page=page, page_size=page_size)


@app.get("/ui/history/{history_id}")
def get_ui_history(history_id: str):
    return _read_history(history_id)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    response_format: str = Form("text"),
    _: None = Depends(require_api_key),
):
    temp_path = await _save_upload_to_temp(file)

    try:
        segments, info = model.transcribe(
            str(temp_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        segment_list = []
        lines = []

        for segment in segments:
            item = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            segment_list.append(item)
            lines.append(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}")

        text_output = "\n".join(lines)
        json_output = {
            "filename": file.filename,
            "language": info.language,
            "segments": segment_list,
            "text": " ".join(item["text"] for item in segment_list),
        }
        _persist_history(file.filename, text_output, json_output)

        if response_format == "json":
            return JSONResponse(json_output)

        return PlainTextResponse(text_output)

    finally:
        temp_path.unlink(missing_ok=True)
