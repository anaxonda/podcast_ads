import ffmpeg
import os
import sys
import subprocess
import json
from typing import List, Dict
from pathlib import Path
from .utils import parse_timestamp
from rich.console import Console

console = Console()

IS_ANDROID = "com.termux" in os.environ.get("PREFIX", "")

class AudioProcessor:
    def __init__(self):
        pass

    def get_duration(self, input_path: str) -> float:
        try:
            probe = ffmpeg.probe(input_path)
            return float(probe['format']['duration'])
        except ffmpeg.Error as e:
            console.print(f"[red]Error probing file: {e.stderr}[/red]")
            raise

    def transcribe_local(self, input_path: str, model_size: str = "tiny", output_dir: str = ".") -> str:
        """
        Transcribes audio locally using whisper-ctranslate2 (PC) or whisper.cpp (Android).
        Returns the path to the generated JSON transcript file.
        """
        input_p = Path(input_path)
        out_p = Path(output_dir)
        expected_json = out_p / f"{input_p.stem}.json"
        
        # Check if already exists
        if expected_json.exists():
             console.log(f"[yellow]Found existing transcript at {expected_json}, skipping Whisper.[/yellow]")
             return str(expected_json)

        if IS_ANDROID:
            return self._transcribe_android(input_path, model_size, str(expected_json))
        else:
            return self._transcribe_pc(input_path, model_size, output_dir, expected_json)

    def _transcribe_pc(self, input_path, model_size, output_dir, expected_json):
        console.log(f"[cyan]Starting local transcription with Whisper ({model_size})...[/cyan]")
        console.log("[dim]Settings: int8, beam=1, batch=4, vad=True[/dim]")
        
        cmd = [
            "whisper-ctranslate2",
            str(input_path),
            "--model", model_size,
            "--language", "en",
            "--compute_type", "int8",
            "--device", "cpu",
            "--output_format", "json",
            "--output_dir", str(output_dir),
            "--threads", "4",
            "--beam_size", "1",
            "--batched", "True",
            "--batch_size", "4",
            "--vad_filter", "True"
        ]
        
        try:
            with console.status(f"[bold yellow]Transcribing locally...[/bold yellow]", spinner="bouncingBar"):
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if expected_json.exists():
                console.log(f"[green]Transcription complete: {expected_json}[/green]")
                return str(expected_json)
            else:
                raise FileNotFoundError(f"Whisper finished but {expected_json} was not found.")
                
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Whisper Error:[/red] {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def _transcribe_android(self, input_path: str, model_size: str, output_json_path: str) -> str:
        """
        Android-specific transcription using whisper.cpp and ffmpeg pipe.
        """
        base_dir = os.path.expanduser("~/whisper.cpp")
        whisper_bin = os.path.join(base_dir, "main")
        # Map model size to q5_1 quantized models which are good for mobile
        model_map = {
            "tiny": "ggml-tiny.en-q5_1.bin",
            "small": "ggml-small.en-q5_1.bin",
            "base": "ggml-base.en-q5_1.bin",
            "medium": "ggml-medium.en-q5_0.bin" # Medium might be heavy
        }
        model_fname = model_map.get(model_size, "ggml-tiny.en-q5_1.bin")
        model_path = os.path.join(base_dir, "models", model_fname)

        if not os.path.exists(whisper_bin):
            raise FileNotFoundError(f"Whisper binary not found at {whisper_bin}. Check Termux setup.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download it in whisper.cpp/models.")

        console.log(f"[cyan]Starting Android transcription (whisper.cpp {model_size})...[/cyan]")

        # 1. FFmpeg Pipe Command
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "-"
        ]

        # 2. Whisper Command
        # -oj means output JSON to stdout? No, --output-json prints to stdout usually or file?
        # The reference implementation captured stdout.
        whisper_cmd = [
            whisper_bin,
            "-m", model_path,
            "-t", "4",
            "-f", "-",            # Read from stdin
            "--no-timestamps",    # Clean console output (we want JSON)
            "--output-json"       # JSON format
        ]

        try:
            with console.status("[bold yellow]Transcribing on Android (Pipe)...[/bold yellow]"):
                # Chain processes
                p1 = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                p2 = subprocess.Popen(whisper_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                p1.stdout.close()  # Allow p1 to receive SIGPIPE if p2 exits
                
                stdout, stderr = p2.communicate()

            if p2.returncode != 0:
                raise RuntimeError(f"Whisper.cpp failed: {stderr}")

            # 3. Normalize JSON
            # Find the JSON start (skip headers)
            json_start = stdout.find('{')
            if json_start == -1:
                raise ValueError("No JSON found in whisper output")
            
            raw_data = json.loads(stdout[json_start:])
            
            # Convert whisper.cpp format to our standard format
            # whisper.cpp: { "transcription": [ { "timestamps": { "from": "...", "to": "..." }, "text": "..." } ] }
            # We need: { "segments": [ { "start": 0.0, "end": 1.0, "text": "..." } ] }
            
            segments = []
            for item in raw_data.get("transcription", []):
                t_from = item.get("timestamps", {}).get("from", "00:00:00,000")
                t_to = item.get("timestamps", {}).get("to", "00:00:00,000")
                
                # Parse timestamps if they are strings
                start = parse_timestamp(t_from.replace(',', '.'))
                end = parse_timestamp(t_to.replace(',', '.'))
                
                segments.append({
                    "start": start,
                    "end": end,
                    "text": item.get("text", "").strip()
                })
            
            final_data = {"segments": segments}
            
            with open(output_json_path, "w") as f:
                json.dump(final_data, f, indent=2)
                
            console.log(f"[green]Android Transcription complete: {output_json_path}[/green]")
            return output_json_path

        except Exception as e:
            console.print(f"[red]Android Transcription Error: {e}[/red]")
            raise

    def cut_and_merge(self, input_path: str, output_path: str, remove_segments: List[Dict]):
        """
        Cuts out the 'remove_segments' and merges the remaining parts.
        """
        total_duration = self.get_duration(input_path)
        
        if not remove_segments:
            console.log("[green]No segments to remove. Skipping processing.[/green]")
            return
        
        # 1. Convert remove segments to "Keep Segments"
        # Sort remove segments by start time
        remove_segments.sort(key=lambda x: parse_timestamp(x['start']))
        
        keep_segments = []
        current_time = 0.0
        
        for seg in remove_segments:
            start_remove = parse_timestamp(seg['start'])
            end_remove = parse_timestamp(seg['end'])
            
            # Sanity checks
            if start_remove >= end_remove:
                console.log(f"[yellow]Skipping invalid segment: {seg['start']} -> {seg['end']} (Start >= End)[/yellow]")
                continue
            if start_remove >= total_duration:
                console.log(f"[yellow]Skipping segment outside audio duration: {seg['start']}[/yellow]")
                continue
                
            # Clamp end time
            end_remove = min(end_remove, total_duration)
            
            if start_remove > current_time:
                keep_segments.append((current_time, start_remove))
            
            current_time = max(current_time, end_remove)
            
        # Add final segment if there's audio left
        if current_time < total_duration:
            keep_segments.append((current_time, total_duration))
            
        if not keep_segments:
            console.print("[yellow]No content segments found to keep! Check your remove logic.[/yellow]")
            return

        console.log(f"Constructing ffmpeg command for {len(keep_segments)} segments...")
        
        # 2. Build FFmpeg filter complex
        input_stream = ffmpeg.input(input_path)
        streams = []
        
        for start, end in keep_segments:
            # Create a trim for each segment
            # atrim operates on audio
            trim = input_stream.filter('atrim', start=start, end=end).filter('asetpts', 'PTS-STARTPTS')
            streams.append(trim)
            
        # 3. Concatenate
        try:
            joined = ffmpeg.concat(*streams, v=0, a=1)
            output = ffmpeg.output(joined, output_path)
            
            console.log("[cyan]Starting audio processing (cutting & merging)...[/cyan]")
            output.run(overwrite_output=True, quiet=True)
            console.print(f"[green]Successfully created {output_path}[/green]")
            
        except ffmpeg.Error as e:
            console.print(f"[red]FFmpeg error: {e.stderr.decode()}[/red]")
            raise
