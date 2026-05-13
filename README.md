# stt-local

Local speech-to-text app using `faster-whisper` with:
- file transcription script (`transcribe.py`)
- FastAPI server (`server.py`)
- Web UI (`/ui`) with upload + transcription progress
- async job endpoints (`/transcribe-jobs`, `/jobs/{job_id}`)

## 1) Install

```bash
./install.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run API + Web UI

```bash
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 9000
```

Open:
- `http://127.0.0.1:9000/ui`
- `http://127.0.0.1:9000/docs`

If port 9000 is busy:

```bash
uvicorn server:app --host 127.0.0.1 --port 9001
```

## 3) Local file transcription (no API)

Put files in `media/`, then run:

```bash
source .venv/bin/activate
python transcribe.py
```

Output files are saved in `transcripts/`.

## 4) Async API usage

Create job:

```bash
curl -X POST http://127.0.0.1:9000/transcribe-jobs \
  -F "file=@media/sample.mp3" \
  -F "response_format=json"
```

Poll status:

```bash
curl http://127.0.0.1:9000/jobs/<job_id>
```

## Notes

- First run downloads the Whisper model (`base.en`).
- Current model/device: CPU with `int8`.
