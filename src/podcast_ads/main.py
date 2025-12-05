import os
import typer
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os
import typer
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from .ai_engine import AIEngine
from .processor import AudioProcessor
from .utils import parse_timestamp
import json

# Load environment variables
load_dotenv()

app = typer.Typer()
console = Console()

@app.command()
def process(
    input_path_str: str = typer.Argument(..., help="Path to input file OR directory of files", metavar="INPUT_PATH"),
    output_dir: str = typer.Option("./output", help="Directory to save processed files"),
    model_size: str = typer.Option("tiny", help="Whisper model size (tiny, small, medium)"),
    dry_run: bool = typer.Option(False, help="Analyze only, do not cut audio"),
    api_key: Optional[str] = typer.Option(None, help="Gemini API Key (overrides .env)")
):
    """
    Process podcast episodes: detect ads/intros/outros using local Whisper transcription + Gemini analysis, 
    strip them out, and generate clean transcripts. Supports single files or batch directory processing.
    """
    
    # 1. Setup
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        console.print("[red]Error: GEMINI_API_KEY not found in .env or arguments.[/red]")
        raise typer.Exit(code=1)
        
    input_path = Path(input_path_str)
    if not input_path.exists():
        console.print(f"[red]Error: Input {input_path} not found.[/red]")
        raise typer.Exit(code=1)
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Discover Files
    files_to_process = []
    if input_path.is_dir():
        extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.opus', '*.ogg']
        for ext in extensions:
            files_to_process.extend(sorted(input_path.rglob(ext)))
        # Remove duplicates if any
        files_to_process = sorted(list(set(files_to_process)))
        console.print(f"[green]Found {len(files_to_process)} audio files in directory.[/green]")
    else:
        files_to_process = [input_path]

    # 3. Initialize Engines
    ai = AIEngine(key)
    processor = AudioProcessor()

    # 4. Process Loop
    success_count = 0
    fail_count = 0

    for current_file in files_to_process:
        console.rule(f"[bold blue]Processing {current_file.name} ({files_to_process.index(current_file)+1}/{len(files_to_process)})[/bold blue]")
        
        try:
            process_single_file(current_file, out_path, ai, processor, model_size, dry_run)
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]Failed to process {current_file.name}: {e}[/bold red]")
            fail_count += 1
            continue

    console.rule("[bold green]Batch Complete[/bold green]")
    console.print(f"Processed: {success_count} | Failed: {fail_count}")

def process_single_file(input_path: Path, out_path: Path, ai: AIEngine, processor: AudioProcessor, model_size: str, dry_run: bool):
    # Check for cached analysis
    cache_file = out_path / f"{input_path.stem}_analysis.json"
    analysis = None
    
    if cache_file.exists():
        console.print(f"[yellow]Found cached analysis at {cache_file}. Using it.[/yellow]")
        try:
            with open(cache_file, 'r') as f:
                analysis = json.load(f)
        except:
            console.print("[red]Cache invalid, re-running AI...[/red]")

    if not analysis:
        # Step 1: Local Transcription (Whisper)
        transcript_json_path = processor.transcribe_local(
            str(input_path), 
            model_size=model_size,
            output_dir=str(out_path)
        )
        
        # Step 2: Semantic Analysis (Gemini)
        analysis = ai.analyze_transcript(transcript_json_path)
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(analysis, f, indent=2)

    segments_to_remove = analysis.get("segments_to_remove", [])
    
    # Reconstruct transcript from segments list
    transcript_segments = analysis.get("transcript_segments", [])
    transcript = ""
    if transcript_segments:
        for segment in transcript_segments:
            label = segment.get("speaker_label", "")
            text = segment.get("text", "")
            transcript += f"{label} {text}\n\n"
    else:
        # Fallback if old format or error
        transcript = analysis.get("transcript", "")
    
    console.print("\n[bold]Identified Segments to Remove:[/bold]")
    for seg in segments_to_remove:
        console.print(f" - [red]{seg['type'].upper()}[/red]: {seg['start']} -> {seg['end']}")
        
    # 3. Audio Processing
    if not dry_run:
        clean_audio_path = out_path / f"{input_path.stem}_clean{input_path.suffix}"
        
        if segments_to_remove:
            processor.cut_and_merge(str(input_path), str(clean_audio_path), segments_to_remove)
        else:
            console.print("[green]No segments to remove. Copying original.[/green]")
            # In a real scenario we might just copy the file
            import shutil
            shutil.copy2(input_path, clean_audio_path)
            console.print(f"[green]Copied to {clean_audio_path}[/green]")
            
        # 4. Transcript Saving
        clean_transcript_path = out_path / f"{input_path.stem}_transcript.md"
        with open(clean_transcript_path, "w") as f:
            f.write(f"# Transcript: {input_path.name}\n\n")
            f.write(transcript)
        console.print(f"[green]Transcript saved to {clean_transcript_path}[/green]")
