# Podcast Ad Stripper

An advanced, cross-platform tool to automatically detect and remove ads, intros, and outros from podcasts and YouTube videos.

Powered by **Whisper** (local transcription) and **Gemini/OpenRouter** (semantic analysis).

## Features

*   **Universal Input:** Processes local audio files (`.mp3`, `.wav`), YouTube URLs, and direct media links.
*   **Smart Detection:** Uses Whisper for accurate text and Gemini 1.5 Pro/Flash (or Grok via OpenRouter) to semantically identify ad breaks.
*   **SponsorBlock Integration:** Prioritizes crowdsourced ad segments for YouTube videos for instant results.
*   **Modular Output:**
    *   **Play Mode:** Stream content ad-free using MPV without downloading/cutting.
    *   **Clean Audio:** Generate a spliced, ad-free MP3.
    *   **Transcripts:** Generate cleaned Markdown transcripts and SRT subtitles.
*   **Android Support:** Fully optimized for Termux with `whisper.cpp` and `mpvKt` integration.

## Installation

### Prerequisites
*   **Python 3.10+**
*   **[uv](https://github.com/astral-sh/uv)** (Recommended package manager)
*   **FFmpeg**
*   **MPV** (For play mode)

### Desktop (Linux/macOS/Windows)
```bash
# Clone the repo
git clone https://github.com/anaxonda/podcast_ads.git
cd podcast_ads

# Install dependencies (including Whisper engine)
uv sync --extra pc --extra google
```

### Android (Termux)
1.  Install Termux packages:
    ```bash
    pkg update
    pkg install python rust binutils build-essential git ffmpeg
    termux-setup-storage
    ```
2.  Install project:
    ```bash
    git clone https://github.com/anaxonda/podcast_ads.git
    cd podcast_ads
    uv sync  # No extras needed (uses whisper.cpp)
    ```
3.  **Download Whisper Model:**
    Create `~/whisper.cpp/models` and download `ggml-tiny.en-q5_1.bin` (or small/base) from [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

## Configuration

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```ini
GEMINI_API_KEY=AIzaSy...
# Optional: For fallback or Grok
OPENROUTER_API_KEY=sk-or... 
AI_MODEL_ORDER=gemini-pro-latest,gemini-2.5-flash,openrouter/x-ai/grok-4.1-fast
```

## Usage

### Basic (Analyze Only)
Generates JSON analysis and a Lua skip script for MPV.
```bash
uv run run.py episode.mp3
```

### Stream Ad-Free (YouTube)
Fetches metadata and launches MPV with auto-skips. No permanent download.
```bash
uv run run.py "https://www.youtube.com/watch?v=VIDEO_ID" --play
```

### Download Clean Audio
Downloads, detects ads, cuts them out, and saves a clean MP3.
```bash
uv run run.py "https://www.youtube.com/watch?v=VIDEO_ID" --save-clean-audio
```

### Generate Transcripts & Subs
```bash
uv run run.py episode.mp3 --save-transcript --save-subs
```

### Batch Processing
Process all files in a directory:
```bash
uv run run.py ./downloads/ --save-clean
```

### CLI Reference

| Flag | Description |
| :--- | :--- |
| `INPUT` | **Required.** Path to a local file, directory, or URL (YouTube/MP3). |
| `--output-dir` | Directory to save all outputs. Defaults to `./output` (PC) or `/sdcard/Download/PodcastAds` (Android). |
| `--model-size` | Whisper model size: `tiny`, `small`, `medium`. Default: `tiny` (fastest). |
| `--api-key` | Override `GEMINI_API_KEY` from environment. |
| **Actions** | |
| `--play` | Stream the media in MPV with ads auto-skipped (Video mode). |
| `--play-audio` | Stream the media in MPV with ads auto-skipped (Audio-only mode). |
| `--save-clean` | Save a new media file with ads removed (Matches input format). |
| `--save-clean-audio`| Save a new MP3 file with ads removed (Converts video to audio if needed). |
| `--save-transcript` | Save a cleaned Markdown transcript (`_transcript.md`). |
| `--save-subs` | Save a cleaned SRT subtitle file (`_clean.srt`). |
| `--dry-run` | Analyze only. Generates `_analysis.json` and `_skips.lua` but skips cutting/saving media. |

## Android Integration

This tool integrates with **mpvKt** on Android for a seamless "Share to Skip" experience.

1.  **Install mpvKt:** Configure its "Config Directory" to `/sdcard/Videos/mpv_config`.
2.  **Setup Script:** Run the helper to create the Share Menu.
    ```bash
    bash android/setup_termux_opener.sh
    ```
3.  **Usage:** Share a YouTube link or File to **Termux**. A menu will appear asking if you want to Play (Video/Audio) or Process.

## How It Works
1.  **Transcribe:** Converts audio to text (Whisper).
2.  **Analyze:** Sends text chunks to LLM to find "Sponsored by" segments.
3.  **Cut/Skip:** Uses timestamps to either slice the file (FFmpeg) or generate a seek script (MPV).