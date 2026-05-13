from pathlib import Path

from faster_whisper import WhisperModel

INPUT_DIR = Path("media")
OUTPUT_DIR = Path("transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}

# For 2 vCPU / 4 GB RAM, start with base.en.
# Try small.en later for better accuracy.
model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
    cpu_threads=2,
    num_workers=1,
)

files = [
    file_path
    for file_path in sorted(INPUT_DIR.iterdir())
    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED
]

if not files:
    print("No supported media files found in media/")

for file_path in files:
    print(f"Transcribing: {file_path.name}")

    segments, info = model.transcribe(
        str(file_path),
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    output_path = OUTPUT_DIR / f"{file_path.stem}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"File: {file_path.name}\n")
        f.write(f"Detected language: {info.language}\n\n")

        for segment in segments:
            line = f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}"
            print(line)
            f.write(line + "\n")

    print(f"Saved: {output_path}")
