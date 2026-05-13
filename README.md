# stt-local

Local speech-to-text project using `faster-whisper` (CPU `int8`) with:
- local file transcription script (`transcribe.py`)
- FastAPI server (`server.py`)
- browser UI (`/ui`) with separate upload and transcription progress
- async job API (`/transcribe-jobs`, `/jobs/{job_id}`)

## Project structure

```text
stt-local/
├── server.py
├── transcribe.py
├── requirements.txt
├── install.sh
├── media/         # put input files here for local script mode
└── transcripts/   # output text files from local script mode
```

## 1) Prerequisites

Install these first:
- Python 3.10+ (tested with Python 3.12)
- `ffmpeg`
- Internet access for first model download (`base.en`)

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ffmpeg
```

Check versions:

```bash
python3 --version
ffmpeg -version
```

## 2) Fresh install (recommended)

From project root:

```bash
cd stt-local
./install.sh
```

What this does:
1. Creates virtual environment `.venv`
2. Upgrades `pip`
3. Installs all dependencies from `requirements.txt`

Manual alternative:

```bash
cd stt-local
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Run the web app (API + UI)

Start server:

```bash
cd stt-local
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 9000
```

Open in browser:
- UI: `http://127.0.0.1:9000/ui`
- Swagger docs: `http://127.0.0.1:9000/docs`
- Health check: `http://127.0.0.1:9000/`

If port `9000` is busy:

```bash
uvicorn server:app --host 127.0.0.1 --port 9001
```

Then use `http://127.0.0.1:9001/ui`.

## 4) First-time model download behavior

On the first transcription request, `faster-whisper` downloads model `base.en` and caches it locally.
- First run is slower
- Later runs are faster (cached model)

## 5) Use the browser UI

1. Open `/ui`
2. Choose audio/video file
3. Choose output format (`text` or `json`)
4. Click **Transcribe**
5. Track:
   - Upload progress bar
   - Transcription progress bar (polled from async job status)

## 6) Local script mode (no server)

Put files into `media/`, for example:
- `media/sample.mp3`

Run:

```bash
cd stt-local
source .venv/bin/activate
python transcribe.py
```

Output is written to `transcripts/<filename>.txt`.

## 7) API usage

### A) Async endpoint (recommended for large files)

Create job:

```bash
curl -X POST http://127.0.0.1:9000/transcribe-jobs \
  -F "file=@media/sample.mp3" \
  -F "response_format=json"
```

Example response:

```json
{"job_id":"<id>","status":"queued"}
```

Poll status:

```bash
curl http://127.0.0.1:9000/jobs/<id>
```

Job states:
- `queued`
- `transcribing`
- `done`
- `error`

When `done`, response includes:
- `result_text`
- `result_json`

### B) Direct blocking endpoint (simple)

```bash
curl -X POST http://127.0.0.1:9000/transcribe \
  -F "file=@media/sample.mp3" \
  -F "response_format=text"
```

Note: this endpoint returns only after transcription completes.

## 8) Reinstall on another computer

1. Copy/pull this folder
2. Install prerequisites (`python3`, `python3-venv`, `ffmpeg`)
3. Run:

```bash
cd stt-local
./install.sh
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 9000
```

4. Open `/ui`

## 9) Updating dependencies

If you add/change packages:

```bash
source .venv/bin/activate
pip install <new-package>
pip freeze > requirements.txt
```

## 10) Troubleshooting

### `address already in use` on port 9000

Use another port:

```bash
uvicorn server:app --host 127.0.0.1 --port 9001
```

### `Method Not Allowed` on `/transcribe`

`/transcribe` is `POST` only. Use:
- `/ui`, or
- `curl -X POST ...`

### Slow first transcription

Expected on first run due to model download and warm-up.

### `ffmpeg` not found

Install it and retry:

```bash
sudo apt-get install -y ffmpeg
```

### Low RAM / slow CPU

Current config is optimized for smaller machines (`base.en`, `cpu`, `int8`).
For better accuracy (slower), switch model in code from `base.en` to `small.en`.
