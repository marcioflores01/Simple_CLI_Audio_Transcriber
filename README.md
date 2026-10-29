# Simple CLI Audio Transcriber

One command turns any audio or video file into a transcript, using OpenAI.

```bash
python3 ~/audio_to_transcript.py
```

Drag the recording into Terminal when asked, press Return, and wait. That's it.

## What it does, in one shot

1. **Converts and splits** the file with ffmpeg: any format (M4A, MP3, WAV, MP4, MOV, QTA…) becomes 16 kHz mono, cut into 10-minute parts.
2. **Transcribes 3 parts at a time** with OpenAI, retrying a part up to 3 times if OpenAI has a hiccup.
3. **Writes the transcript** next to the recording, in `<name>_Transcription_Output/`:

| File | Contents |
|---|---|
| `<name>_TRANSCRIPT.txt` | Plain text with `[hh:mm:ss]` timestamps |
| `<name>_TRANSCRIPT.rtf` | Same, opens in Word/Pages (accents display correctly) |
| `<name>_TRANSCRIPT.srt` | Subtitles for the whole recording (fast mode only) |

If a part still fails, the parts that worked are saved as `_PARTIAL_TRANSCRIPT`.

## Options

```bash
python3 ~/audio_to_transcript.py "meeting.m4a"          # pass the file directly
python3 ~/audio_to_transcript.py --speakers             # label who is speaking (Speaker A, B…)
```

| Mode | Model | Speed | Speaker labels |
|---|---|---|---|
| default | `whisper-1` | fast | no |
| `--speakers` | `gpt-4o-transcribe-diarize` | about 3–5× slower | yes (letters can change between 10-minute parts) |

## Setup (once)

```bash
brew install ffmpeg
pip3 install openai
```

**API key:** the first run asks for your OpenAI key and saves it in the macOS Keychain (service `openai-api-key`). It is never written into the script. To change it later:

```bash
security delete-generic-password -s openai-api-key
```

The next run will ask again. An `OPENAI_API_KEY` environment variable, if set, takes priority.

## Installing the command

The script lives in this repo. `~/audio_to_transcript.py` is a symbolic link to it, so there is one real copy:

```bash
ln -s ~/Developer/Simple_CLI_Audio_Transcriber/audio_to_transcript.py ~/audio_to_transcript.py
```
