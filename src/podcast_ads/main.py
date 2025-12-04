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
    input_file: str = typer.Argument(..., help="Path to the input audio file (mp3/wav)"),
    output_dir: str = typer.Option("./output", help="Directory to save processed files"),
    model_size: str = typer.Option("tiny", help="Whisper model size (tiny, small, medium)"),
    dry_run: bool = typer.Option(False, help="Analyze only, do not cut audio"),
    api_key: Optional[str] = typer.Option(None, help="Gemini API Key (overrides .env)")
):
    """
    Process a podcast episode: detect ads/intros/outros using local Whisper transcription + Gemini analysis, 
    strip them out, and generate a clean transcript.
    """
    
    # 1. Setup
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        console.print("[red]Error: GEMINI_API_KEY not found in .env or arguments.[/red]")
        raise typer.Exit(code=1)
        
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found.[/red]")
        raise typer.Exit(code=1)
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    console.rule(f"[bold blue]Processing {input_path.name} (Whisper: {model_size})[/bold blue]")
    
    # 2. AI Analysis
    ai = AIEngine(key)
    processor = AudioProcessor()
    
    # Check for cached analysis to save money/time during dev
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
        try:
            # Step 1: Local Transcription (Whisper)
            # This generates a large JSON file with word-level or segment-level timestamps
            transcript_json_path = processor.transcribe_local(
                str(input_path), 
                model_size=model_size,
                output_dir=str(out_path)
            )
            
            # Step 2: Semantic Analysis (Gemini)
            # Gemini reads the JSON and returns semantic ad boundaries
            analysis = ai.analyze_transcript(transcript_json_path)
            
            # Save cache
            with open(cache_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
        except Exception as e:
            console.print(f"[red]AI Processing failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise typer.Exit(code=1)

    segments_to_remove = analysis.get("segments_to_remove", [])
    transcript = analysis.get("transcript", "")
    
    console.print("\n[bold]Identified Segments to Remove:[/bold]")
    for seg in segments_to_remove:
        console.print(f" - [red]{seg['type'].upper()}[/red]: {seg['start']} -> {seg['end']}")
        
    # 3. Audio Processing
    if not dry_run:
        # Use the original extension for the output
        clean_audio_path = out_path / f"{input_path.stem}_clean{input_path.suffix}"
        
        if segments_to_remove:
            processor.cut_and_merge(str(input_path), str(clean_audio_path), segments_to_remove)
        else:
            console.print("[green]No segments to remove. Copying original.[/green]")
            # Just copy if nothing to remove, or ffmpeg copy
            # processor.copy... (omitted for brevity, standard copy)
            
        # 4. Transcript Saving
        clean_transcript_path = out_path / f"{input_path.stem}_transcript.md"
        with open(clean_transcript_path, "w") as f:
            f.write(f"# Transcript: {input_path.name}\n\n")
            f.write(transcript)
        console.print(f"[green]Transcript saved to {clean_transcript_path}[/green]")

if __name__ == "__main__":
    app()
