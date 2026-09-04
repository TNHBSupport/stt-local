#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/forge/transcribe.on-forge.com}"
CURRENT_DIR="${APP_ROOT}/current"
VENV_DIR="${APP_ROOT}/shared-venv"
LOG_FILE="${APP_ROOT}/uvicorn.log"
PID_FILE="${APP_ROOT}/uvicorn.pid"
HOST="${STT_HOST:-127.0.0.1}"
PORT="${STT_PORT:-9000}"

if [ -f "${APP_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${APP_ROOT}/.env"
  set +a
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python3" -m pip install --upgrade pip
"${VENV_DIR}/bin/python3" -m pip install -r "${CURRENT_DIR}/requirements.txt"

if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    kill "${OLD_PID}" || true
    sleep 2
  fi
fi

pkill -f "uvicorn server:app --host ${HOST} --port ${PORT}" || true

export PYTHONUNBUFFERED=1
export STT_LOG_FILE="${STT_LOG_FILE:-$LOG_FILE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CT2_USE_EXPERIMENTAL_PACKED_GEMM="${CT2_USE_EXPERIMENTAL_PACKED_GEMM:-0}"

cd "${CURRENT_DIR}"
# Respawn uvicorn after crash (OOM) so Cloudflare is not stuck on 502 until the next deploy.
# PID file stores this wrapper; deploy stop still kill "$(cat PID_FILE)" then pkill uvicorn.
nohup env PYTHONUNBUFFERED=1 STT_LOG_FILE="${STT_LOG_FILE}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  CT2_USE_EXPERIMENTAL_PACKED_GEMM="${CT2_USE_EXPERIMENTAL_PACKED_GEMM}" \
  bash -c '
    child=""
    stopping=0
    shutdown() {
      stopping=1
      if [ -n "${child}" ]; then
        kill "${child}" 2>/dev/null || true
        wait "${child}" 2>/dev/null || true
      fi
      exit 0
    }
    trap shutdown TERM INT
    while [ "${stopping}" -eq 0 ]; do
      "$@" &
      child=$!
      wait "${child}" || true
      child=""
      if [ "${stopping}" -eq 0 ]; then
        echo "uvicorn exited, restarting in 2s" >&2
        sleep 2
      fi
    done
  ' _ "${VENV_DIR}/bin/uvicorn" server:app --host "${HOST}" --port "${PORT}" --timeout-keep-alive 75 \
  > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

for attempt in $(seq 1 15); do
  if curl -fsS "http://${HOST}:${PORT}/" >/dev/null 2>&1; then
    echo "STT server restarted on ${HOST}:${PORT}"
    exit 0
  fi
  sleep 2
done

echo "STT server failed to respond on ${HOST}:${PORT} after deploy" >&2
exit 1
