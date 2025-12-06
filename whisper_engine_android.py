import subprocess
import json
import os
import sys

class WhisperTranscriber:
    def __init__(self,
                 model_name="ggml-tiny.en-q5_1.bin",
                 base_dir="/data/data/com.termux/files/home/whisper.cpp",
                 threads="4"):
        """
        Initializes the Whisper Wrapper for Termux.

        Args:
            model_name (str): Name of the model file (e.g., ggml-tiny.en-q5_1.bin).
            base_dir (str): Absolute path to the whisper.cpp folder.
            threads (str): Number of threads to use (4 is optimal for Exynos 9820).
        """
        # 1. Define Absolute Paths for Binaries
        # Termux standard path for ffmpeg
        self.ffmpeg_bin = "/data/data/com.termux/files/usr/bin/ffmpeg"
        self.whisper_bin = os.path.join(base_dir, "main")
        self.model_path = os.path.join(base_dir, "models", model_name)
        self.threads = threads

        # 2. Validation
        self._validate_paths()

    def _validate_paths(self):
        """Ensures all necessary binaries and models exist."""
        if not os.path.exists(self.ffmpeg_bin):
            raise FileNotFoundError(f"FFmpeg not found at {self.ffmpeg_bin}. Run 'pkg install ffmpeg'")
        if not os.path.exists(self.whisper_bin):
            raise FileNotFoundError(f"Whisper binary not found at {self.whisper_bin}. Did you run 'make'?")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Did you download {os.path.basename(self.model_path)}?")

    def transcribe(self, audio_path, output_format="text"):
        """
        Transcribes an audio file.

        Args:
            audio_path (str): Path to the input audio file.
            output_format (str): 'text' for raw string, 'json' for full data object.

        Returns:
            str or dict: The transcription result.
        """
        # Ensure input path is absolute
        abs_audio_path = os.path.abspath(audio_path)

        if not os.path.exists(abs_audio_path):
            return {"error": f"File not found: {abs_audio_path}"}

        # Construct FFmpeg Command (Audio Pipeline)
        ffmpeg_cmd = [
            self.ffmpeg_bin,
            "-i", abs_audio_path,
            "-ar", "16000",       # Resample to 16k
            "-ac", "1",           # Mono
            "-c:a", "pcm_s16le",  # PCM format
            "-f", "wav",          # WAV container
            "-"                   # Pipe to stdout
        ]

        # Construct Whisper Command
        whisper_cmd = [
            self.whisper_bin,
            "-m", self.model_path,
            "-t", self.threads,
            "-f", "-",            # Read from stdin
            "--no-timestamps",    # Clean output
            "--output-json"       # Structued output
        ]

        try:
            # 1. Start FFmpeg
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            # 2. Start Whisper (fed by FFmpeg)
            whisper_proc = subprocess.Popen(
                whisper_cmd,
                stdin=ffmpeg_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Allow ffmpeg to receive SIGPIPE if whisper exits
            ffmpeg_proc.stdout.close()

            # 3. Get Output
            stdout, stderr = whisper_proc.communicate()

            if whisper_proc.returncode != 0:
                raise RuntimeError(f"Whisper Error: {stderr}")

            # 4. Parse JSON
            # Clean up potential logs before JSON starts
            json_start = stdout.find('{')
            if json_start == -1:
                return stdout.strip() # Fallback to raw text

            clean_json = stdout[json_start:]
            data = json.loads(clean_json)

            if output_format == "json":
                return data
            else:
                # Combine segments for clean text
                full_text = "".join([seg.get('text', '') for seg in data.get('transcription', [])])
                return full_text.strip()

        except Exception as e:
            return {"error": str(e)}

# Allow running this script directly for testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python whisper_engine.py <audio_file>")
        sys.exit(1)

    engine = WhisperTranscriber() # Defaults to tiny.en-q5_1
    print("Transcribing...")
    result = engine.transcribe(sys.argv[1])
    print("\n--- Result ---\n")
    print(result)
