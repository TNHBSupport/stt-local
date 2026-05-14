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

For domain/external usage, set security env vars before starting:

```bash
export STT_API_KEY="replace-with-a-long-random-secret"
export STT_CORS_ORIGINS="https://your-app.com,https://admin.your-app.com"
uvicorn server:app --host 127.0.0.1 --port 9000
```

Open in browser:
- UI: `http://127.0.0.1:9000/ui` (no API key field; uses internal UI routes)
- Swagger docs: `http://127.0.0.1:9000/docs`
- Health check: `http://127.0.0.1:9000/`

If port `9000` is busy:

```bash
uvicorn server:app --host 127.0.0.1 --port 9001
```

Then use `http://127.0.0.1:9001/ui`.

## 4) Deploy on Laravel Forge (Domain + HTTPS)

This is the recommended production flow for your use case.

### 4.1 Create server and site in Forge

1. Provision an Ubuntu server (AWS or DigitalOcean) from Forge.
2. Create a site, for example: `stt-api.yourdomain.com`.
3. Enable SSL in Forge (Let's Encrypt).
4. Point DNS `A` record to the server IP.

### 4.2 Upload project to server

Option A: Git deploy (recommended)

```bash
git clone <your-repo-url> /home/forge/stt-local
cd /home/forge/stt-local
```

Option B: upload tar/zip and extract to `/home/forge/stt-local`.

### 4.3 Install dependencies

```bash
cd /home/forge/stt-local
./install.sh
```

### 4.4 Set environment variables

Add to shell profile or service config:

```bash
export STT_API_KEY="replace-with-a-long-random-secret"
export STT_CORS_ORIGINS="https://your-frontend.com,https://admin.your-frontend.com"
```

### 4.5 Create systemd service (recommended)

Create `/etc/systemd/system/stt-local.service`:

```ini
[Unit]
Description=STT Local FastAPI Service
After=network.target

[Service]
User=forge
Group=forge
WorkingDirectory=/home/forge/stt-local
Environment="STT_API_KEY=replace-with-a-long-random-secret"
Environment="STT_CORS_ORIGINS=https://your-frontend.com,https://admin.your-frontend.com"
ExecStart=/home/forge/stt-local/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 9000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable stt-local
sudo systemctl start stt-local
sudo systemctl status stt-local
```

### 4.6 Configure Nginx reverse proxy (Forge site)

In your Forge site Nginx config, proxy requests to local uvicorn:

```nginx
location / {
    proxy_pass http://127.0.0.1:9000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Tune for large audio/video uploads
    client_max_body_size 1024M;
    proxy_connect_timeout 60s;
    proxy_send_timeout 3600s;
    proxy_read_timeout 3600s;
}
```

Reload Nginx from Forge UI or:

```bash
sudo systemctl reload nginx
```

### 4.7 Verify deployment

```bash
curl https://stt-api.yourdomain.com/
curl -X POST https://stt-api.yourdomain.com/transcribe-jobs \
  -H "X-API-Key: replace-with-a-long-random-secret" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

### 4.8 Auto-deploy and restart on every push

Use this when Forge does not provide a daemon UI. The repo includes `forge-deploy.sh`, which creates/updates the shared Python venv, stops the old Uvicorn process, and starts the new one in the background.

In `/home/forge/transcribe.on-forge.com/.env`, add:

```bash
STT_API_KEY=replace-with-a-long-random-secret
STT_CORS_ORIGINS=https://transcribe.on-forge.com
```

In Forge site Deployment Script, use only this:

```bash
cd /home/forge/transcribe.on-forge.com/current
chmod +x forge-deploy.sh
./forge-deploy.sh
```

Then enable Auto Deploy for the `main` branch. After each push, Forge deploys the new release and runs `forge-deploy.sh`, so the running server reloads the latest code.

To verify after deployment:

```bash
curl http://127.0.0.1:9000/
tail -n 50 /home/forge/transcribe.on-forge.com/uvicorn.log
```

## 5) Deploy on AWS EC2 (without Forge)

### 5.1 Launch and access EC2

1. Launch Ubuntu EC2 instance.
2. Open security group ports:
   - `22` (SSH)
   - `80` (HTTP)
   - `443` (HTTPS)
3. SSH in:

```bash
ssh -i /path/to/key.pem ubuntu@<ec2-public-ip>
```

### 5.2 Install system packages

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ffmpeg nginx certbot python3-certbot-nginx git
```

### 5.3 Deploy app

```bash
git clone <your-repo-url> /opt/stt-local
cd /opt/stt-local
./install.sh
```

### 5.4 Create systemd service

Create `/etc/systemd/system/stt-local.service` (same as Forge section; set `User=ubuntu`, `WorkingDirectory=/opt/stt-local`, and `.venv` path accordingly):

```ini
[Unit]
Description=STT Local FastAPI Service
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/stt-local
Environment="STT_API_KEY=replace-with-a-long-random-secret"
Environment="STT_CORS_ORIGINS=https://your-frontend.com"
ExecStart=/opt/stt-local/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 9000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable stt-local
sudo systemctl start stt-local
```

### 5.5 Nginx domain config

Create `/etc/nginx/sites-available/stt-local`:

```nginx
server {
    server_name stt-api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        client_max_body_size 1024M;
        proxy_connect_timeout 60s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
    }
}
```

Enable config:

```bash
sudo ln -s /etc/nginx/sites-available/stt-local /etc/nginx/sites-enabled/stt-local
sudo nginx -t
sudo systemctl reload nginx
```

Enable HTTPS:

```bash
sudo certbot --nginx -d stt-api.yourdomain.com
```

## 6) Use the browser UI

1. Open `/ui`
2. Choose audio/video file
3. Choose output format (`text` or `json`)
4. Click **Transcribe**
5. Track:
   - Upload progress bar
   - Transcription progress bar (polled from async job status)

## 7) Local script mode (no server)

Put files into `media/`, for example:
- `media/sample.mp3`

Run:

```bash
cd stt-local
source .venv/bin/activate
python transcribe.py
```

Output is written to `transcripts/<filename>.txt`.

## 8) API usage

### A) Async endpoint (recommended for large files)

Create job:

```bash
curl -X POST https://stt-api.yourdomain.com/transcribe-jobs \
  -H "X-API-Key: replace-with-a-long-random-secret" \
  -F "file=@media/sample.mp3" \
  -F "response_format=json"
```

Example response:

```json
{"job_id":"<id>","status":"queued"}
```

Poll status:

```bash
curl -H "X-API-Key: replace-with-a-long-random-secret" \
  https://stt-api.yourdomain.com/jobs/<id>
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
curl -X POST https://stt-api.yourdomain.com/transcribe \
  -H "X-API-Key: replace-with-a-long-random-secret" \
  -F "file=@media/sample.mp3" \
  -F "response_format=text"
```

Note: this endpoint returns only after transcription completes.

## 9) Reinstall on another computer

1. Copy/pull this folder
2. Install prerequisites (`python3`, `python3-venv`, `ffmpeg`)
3. Run:

```bash
cd stt-local
./install.sh
source .venv/bin/activate
export STT_API_KEY="replace-with-a-long-random-secret"
export STT_CORS_ORIGINS="https://your-app.com"
uvicorn server:app --host 127.0.0.1 --port 9000
```

4. Open `/ui`

## 10) Updating dependencies

If you add/change packages:

```bash
source .venv/bin/activate
pip install <new-package>
pip freeze > requirements.txt
```

## 11) Troubleshooting

### `address already in use` on port 9000

Use another port:

```bash
uvicorn server:app --host 127.0.0.1 --port 9001
```

### `Method Not Allowed` on `/transcribe`

`/transcribe` is `POST` only. Use:
- `/ui`, or
- `curl -X POST ...`

### `{"detail":"Invalid or missing API key"}`

Your server has `STT_API_KEY` enabled and request is missing/incorrect `X-API-Key` header.

### Browser call blocked by CORS

Add your frontend origin to `STT_CORS_ORIGINS`, comma-separated:

```bash
export STT_CORS_ORIGINS="https://your-app.com,https://staging.your-app.com"
```

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
