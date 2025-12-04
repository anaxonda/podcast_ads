# Podcast Ad Stripper

A CLI tool powered by Gemini 1.5 to automatically detect and strip ads, intros, and outros from podcast audio files.

## Usage

1. Set your `GEMINI_API_KEY` in `.env`.
2. Run the processor:
   ```bash
   uv run run.py input.mp3 --output-dir ./clean_episodes
   ```
