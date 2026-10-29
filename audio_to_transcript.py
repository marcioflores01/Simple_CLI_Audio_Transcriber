#!/usr/bin/env python3
"""Transcribe any audio/video file in one shot.

    python3 ~/audio_to_transcript.py                  # asks for the file (drag it into Terminal)
    python3 ~/audio_to_transcript.py "meeting.m4a"    # or pass it directly
    python3 ~/audio_to_transcript.py --speakers ...   # label speakers (slower, uses gpt-4o-transcribe-diarize)

Output goes next to the recording, in <name>_Transcription_Output/:
    <name>_TRANSCRIPT.txt   plain text with [hh:mm:ss] timestamps
    <name>_TRANSCRIPT.rtf   same, opens in Word/Pages
    <name>_TRANSCRIPT.srt   subtitles for the whole recording (fast mode only)

The OpenAI key is read from the OPENAI_API_KEY environment variable or the macOS
Keychain. The first run asks for it once and saves it to the Keychain, so the key
never lives in this file.
"""
import argparse
import getpass
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

KEYCHAIN_SERVICE = "openai-api-key"
CHUNK_SECONDS = 600   # 10-minute parts; at 64 kbps mono each is ~5 MB, far under the 25 MB API limit
PARALLEL_PARTS = 3    # parts sent to OpenAI at the same time
MAX_ATTEMPTS = 3      # per part, for timeouts, rate limits and server errors


# ─── API key ────────────────────────────────────────────────────────────────
def get_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    found = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", getpass.getuser(), "-w"],
        capture_output=True, text=True)
    if found.returncode == 0 and found.stdout.strip():
        return found.stdout.strip()

    print("No OpenAI API key found. Paste it below (it won't be shown); it will be saved in your Mac's Keychain.")
    key = getpass.getpass("OpenAI API key: ").strip()
    if not key.startswith("sk-"):
        sys.exit("That doesn't look like an OpenAI key (it should start with sk-).")
    subprocess.run(
        ["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", getpass.getuser(), "-w", key],
        check=True)
    print("✓ Key saved to Keychain.\n")
    return key


# ─── Helpers ────────────────────────────────────────────────────────────────
def clock(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def rtf_escape(text):
    out = []
    for ch in text:
        c = ord(ch)
        if ch in "\\{}":
            out.append("\\" + ch)
        elif c > 127:
            # RTF is ASCII: write accents/emoji as UTF-16 \uN? escapes
            data = ch.encode("utf-16-le")
            for j in range(0, len(data), 2):
                n = int.from_bytes(data[j:j + 2], "little")
                out.append(f"\\u{n - 65536 if n > 32767 else n}?")
        else:
            out.append(ch)
    return "".join(out)


def field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


# ─── Steps ──────────────────────────────────────────────────────────────────
def split_audio(audio_path, work_dir):
    """One ffmpeg pass: decode anything, downmix to 16 kHz mono, cut into 10-minute MP3 parts."""
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k",
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
        str(work_dir / "part_%03d.mp3"),
    ], check=True)
    return sorted(work_dir.glob("part_*.mp3"))


def transcribe_part(client, part, speakers):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with open(part, "rb") as f:
                if speakers:
                    return client.audio.transcriptions.create(
                        model="gpt-4o-transcribe-diarize", file=f,
                        response_format="diarized_json", chunking_strategy="auto")
                return client.audio.transcriptions.create(
                    model="whisper-1", file=f,
                    response_format="verbose_json", timestamp_granularities=["segment"])
        except Exception as e:
            status = getattr(e, "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or attempt == MAX_ATTEMPTS:
                raise
            print(f"   {part.name}: {e.__class__.__name__} — retrying in {attempt * 10}s")
            time.sleep(attempt * 10)


def paragraphs(result, offset, speakers):
    """Return [(start_seconds, end_seconds, label, text)] with times relative to the whole recording."""
    rows = []
    for seg in field(result, "segments") or []:
        text = (field(seg, "text") or "").strip()
        if not text:
            continue
        start, end = offset + (field(seg, "start") or 0), offset + (field(seg, "end") or 0)
        label = f"Speaker {field(seg, 'speaker') or '?'}" if speakers else None
        if speakers and rows and rows[-1][2] == label:
            rows[-1] = (rows[-1][0], end, label, rows[-1][3] + " " + text)
        else:
            rows.append((start, end, label, text))
    if not rows and (field(result, "text") or "").strip():
        rows.append((offset, offset, None, field(result, "text").strip()))
    return rows


def write_outputs(audio_path, out_dir, parts_rows, speakers, complete):
    stem = audio_path.stem
    suffix = "TRANSCRIPT" if complete else "PARTIAL_TRANSCRIPT"
    total = len(parts_rows)

    header = [f"TRANSCRIPT: {audio_path.name}", f"Generated: {datetime.now():%Y-%m-%d %H:%M}"]
    if not complete:
        header.append(f"PARTIAL: {sum(r is not None for r in parts_rows)} of {total} parts transcribed")
    lines = header + ["", "═" * 60, ""]
    for i, rows in enumerate(parts_rows):
        if rows is None:
            continue
        if total > 1:
            lines += [f"───── PART {i + 1} of {total} (from {clock(i * CHUNK_SECONDS)}) ─────", ""]
        for start, _, label, text in rows:
            lines += [f"[{clock(start)}] {label + ': ' if label else ''}{text}", ""]
    txt = "\n".join(lines)

    rtf_body = []
    for line in txt.split("\n"):
        if line.startswith("[") and "] Speaker " in line and ":" in line[12:]:
            stamp, rest = line[:11], line[11:]
            label, text = rest.split(":", 1)
            # the space after \\b0 only ends the control word; the text keeps its own leading space
            rtf_body.append(f"{rtf_escape(stamp)}\\b {rtf_escape(label)}:\\b0 {rtf_escape(text)}\\par")
        else:
            rtf_body.append(rtf_escape(line) + "\\par")
    rtf = "{\\rtf1\\ansi\\deff0\n{\\fonttbl{\\f0 Arial;}}\n\\f0\\fs22\n" + "\n".join(rtf_body) + "\n}"

    written = []
    for ext, content in (("txt", txt), ("rtf", rtf)):
        path = out_dir / f"{stem}_{suffix}.{ext}"
        path.write_text(content, encoding="utf-8")
        written.append(path)

    if not speakers:
        cues, n = [], 1
        for rows in parts_rows:
            for start, end, _, text in rows or []:
                cues.append(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
                n += 1
        path = out_dir / f"{stem}_{suffix}.srt"
        path.write_text("\n".join(cues), encoding="utf-8")
        written.append(path)
    return written


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio/video file into TXT + RTF (+ SRT).")
    parser.add_argument("file", nargs="?", help="audio or video file (M4A, MP3, WAV, MP4, MOV, QTA…)")
    parser.add_argument("--speakers", action="store_true",
                        help="label who is speaking (slower; gpt-4o-transcribe-diarize)")
    args = parser.parse_args()

    raw = args.file or input("Drag your audio file here (or paste its path), then press Return: ")
    audio_path = Path(raw.strip().strip('"').strip("'").replace("\\ ", " ")).expanduser()
    if not audio_path.is_file():
        sys.exit(f"File not found: {audio_path}")

    client = OpenAI(api_key=get_api_key())
    out_dir = audio_path.parent / f"{audio_path.stem}_Transcription_Output"
    out_dir.mkdir(exist_ok=True)
    started = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"\n=== Step 1: Converting and splitting {audio_path.name} ===")
        parts = split_audio(audio_path, Path(tmp))
        print(f"   {len(parts)} part(s) of up to {CHUNK_SECONDS // 60} min")

        mode = "with speaker labels" if args.speakers else "fast mode (whisper-1)"
        print(f"\n=== Step 2: Transcribing {len(parts)} part(s), {PARALLEL_PARTS} at a time, {mode} ===")
        parts_rows = [None] * len(parts)
        failed = None
        with ThreadPoolExecutor(max_workers=PARALLEL_PARTS) as pool:
            futures = {pool.submit(transcribe_part, client, p, args.speakers): i for i, p in enumerate(parts)}
            for future, i in futures.items():
                try:
                    parts_rows[i] = paragraphs(future.result(), i * CHUNK_SECONDS, args.speakers)
                    print(f"   ✓ Part {i + 1} of {len(parts)}")
                except Exception as e:
                    failed = failed or e
                    print(f"   ✗ Part {i + 1} of {len(parts)} failed: {e}")

    complete = failed is None
    if not complete and not any(r is not None for r in parts_rows):
        sys.exit(f"\nTranscription failed: {failed}")

    print("\n=== Step 3: Writing transcript files ===")
    for path in write_outputs(audio_path, out_dir, parts_rows, args.speakers, complete):
        print(f"   {path}")

    minutes = (time.time() - started) / 60
    if complete:
        print(f"\nALL DONE in {minutes:.1f} min.")
    else:
        sys.exit(f"\nFinished with errors in {minutes:.1f} min — saved the parts that worked (PARTIAL_TRANSCRIPT).")


if __name__ == "__main__":
    main()
