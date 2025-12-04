import ffmpeg
import os
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

    def create_chunked_proxies(self, input_path: str, chunk_minutes: int = 10) -> List[tuple[str, float]]:
        """
        Creates lightweight MP3 proxy chunks (mono, 64k).
        Returns a list of (file_path, start_offset_seconds) tuples.
        """
        input_p = Path(input_path)
        chunk_seconds = chunk_minutes * 60
        
        # Pattern for ffmpeg output: .tmp_proxy_stem_000.mp3
        output_pattern = input_p.parent / f".tmp_ai_proxy_{input_p.stem}_%03d.mp3"
        
        # Clean up previous runs if any
        for existing in input_p.parent.glob(f".tmp_ai_proxy_{input_p.stem}_*.mp3"):
            existing.unlink()

        console.log(f"[cyan]Splitting {input_p.name} into {chunk_minutes}-minute proxy chunks...[/cyan]")
        
        try:
            with console.status("[bold yellow]FFmpeg is segmenting audio (this might take a minute)...[/bold yellow]", spinner="bouncingBar"):
                (
                    ffmpeg
                    .input(input_path)
                    .output(
                        str(output_pattern), 
                        f="segment", 
                        segment_time=chunk_seconds, 
                        reset_timestamps=1,
                        ac=1, 
                        ar=44100, 
                        audio_bitrate="64k"
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            
            # Collect results
            chunks = []
            found_files = sorted(input_p.parent.glob(f".tmp_ai_proxy_{input_p.stem}_*.mp3"))
            
            for i, f_path in enumerate(found_files):
                offset = i * chunk_seconds
                chunks.append((str(f_path), offset))
                
            console.log(f"[green]Created {len(chunks)} chunks for processing.[/green]")
            return chunks
            
        except ffmpeg.Error as e:
            console.print(f"[red]Error creating proxy chunks: {e.stderr.decode()}[/red]")
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
