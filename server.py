import tempfile
import threading
import uuid
import os
from pathlib import Path
from typing import Annotated

import av
from fastapi import Depends, FastAPI, File, Form, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel

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


async def _create_job(file: UploadFile, response_format: str) -> dict:
    suffix = Path(file.filename or "audio").suffix or ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp_path = Path(temp.name)
        temp.write(await file.read())

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
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 12px 34px rgba(11, 33, 63, 0.1);
    }
    h1 { margin: 0 0 8px; letter-spacing: 0.2px; }
    .muted { color: var(--muted); margin: 0 0 18px; }
    .row { margin: 14px 0; }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr 180px auto auto;
      align-items: center;
    }
    input[type=file], select {
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
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .statusline { margin: 10px 0 0; color: var(--muted); font-size: 14px; min-height: 20px; }
    .progress-wrap { margin-top: 14px; }
    .progress-label { font-size: 13px; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
    .download-actions { display: flex; gap: 10px; margin-top: 14px; }
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
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>Local Speech-to-Text</h1>
      <p class=\"muted\">Upload audio/video and track progress while transcribing.</p>
      <p class=\"muted\"><strong>Deploy Check:</strong> Auto deploy marker v2</p>

      <form id=\"upload-form\">
        <div class=\"row grid\">
          <input id=\"file\" name=\"file\" type=\"file\" accept=\"audio/*,video/*\" required />

          <select id=\"format\" name=\"response_format\">
            <option value=\"text\" selected>text</option>
            <option value=\"json\">json</option>
          </select>

          <button id=\"submit-btn\" type=\"submit\">Transcribe</button>
          <button id=\"cancel-btn\" class=\"cancel-button\" type=\"button\" disabled>Cancel</button>
          <button id=\"reset-btn\" type=\"button\">Reset</button>
        </div>
      </form>

      <p id=\"status\" class=\"statusline\"></p>

      <div class=\"progress-wrap\">
        <div class=\"progress-label\"><span>Upload</span><span id=\"upload-pct\">0%</span></div>
        <div class=\"progress\"><div id=\"upload-bar\" class=\"bar\"></div></div>
      </div>

      <div class=\"progress-wrap\">
        <div class=\"progress-label\"><span>Transcription</span><span id=\"transcribe-pct\">0%</span></div>
        <div class=\"progress\"><div id=\"transcribe-bar\" class=\"bar\"></div></div>
      </div>

      <div class=\"download-actions\">
        <button id=\"download-text-btn\" type=\"button\" disabled>Download TXT</button>
        <button id=\"download-json-btn\" type=\"button\" disabled>Download JSON</button>
      </div>

      <pre id=\"result\"></pre>
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
            if (format === 'json') {
              result.textContent = JSON.stringify(lastResultJson, null, 2);
            } else {
              result.textContent = lastResultText;
            }
            submitBtn.disabled = false;
            cancelBtn.disabled = true;
            downloadTextBtn.disabled = !lastResultText;
            downloadJsonBtn.disabled = !lastResultJson;
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
      const format = document.getElementById('format').value;

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


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    response_format: str = Form("text"),
    _: None = Depends(require_api_key),
):
    suffix = Path(file.filename).suffix or ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp_path = Path(temp.name)
        temp.write(await file.read())

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

        if response_format == "json":
            return JSONResponse(
                {
                    "filename": file.filename,
                    "language": info.language,
                    "segments": segment_list,
                    "text": " ".join(item["text"] for item in segment_list),
                }
            )

        return PlainTextResponse("\n".join(lines))

    finally:
        temp_path.unlink(missing_ok=True)
