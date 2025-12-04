import ffmpeg
import os
import subprocess
from typing import List, Dict
from pathlib import Path
from .utils import parse_timestamp
from rich.console import Console

console = Console()

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
        Transcribes audio locally using whisper-ctranslate2.
        Returns the path to the generated JSON transcript file.
        """
        input_p = Path(input_path)
        out_p = Path(output_dir)
        expected_json = out_p / f"{input_p.stem}.json"
        
        # Check if already exists
        if expected_json.exists():
             console.log(f"[yellow]Found existing transcript at {expected_json}, skipping Whisper.[/yellow]")
             return str(expected_json)

        console.log(f"[cyan]Starting local transcription with Whisper ({model_size})...[/cyan]")
        
        # cmd = [
        #     "whisper-ctranslate2",
        #     str(input_path),
        #     "--model", model_size,
        #     "--compute_type", "int8",
        #     "--device", "cpu",
        #     "--output_format", "json",
        #     "--output_dir", str(output_dir),
        #     "--threads", str(os.cpu_count() or 4) # Maximize CPU usage
        # ]
        cmd = [
    "whisper-ctranslate2",
    str(input_path),
    "--model", "tiny",
    "--language", "en",           # Specify language to skip detection
    "--compute_type", "int8",     # CPU Optimized
    "--device", "cpu",
    "--threads", "4", # USE PHYSICAL CORES ONLY
    "--beam_size", "1",           # Greedy decoding (Faster)
    "--batched", "True",          # Enable batching
    "--batch_size", "8",          # Process 8 chunks at once
    "--vad_filter", "True",       # Skip silent parts
    "--output_format", "json",
    "--output_dir", str(output_dir)
]
        
        try:
            with console.status(f"[bold yellow]Transcribing locally... (this will take several minutes)[/bold yellow]", spinner="bouncingBar"):
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if expected_json.exists():
                console.log(f"[green]Transcription complete: {expected_json}[/green]")
                return str(expected_json)
            else:
                raise FileNotFoundError(f"Whisper finished but {expected_json} was not found.")
                
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Whisper Error:[/red] {e.stderr.decode() if e.stderr else str(e)}")
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
