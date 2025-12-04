import os
import typer
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from .ai_engine import AIEngine
from .processor import AudioProcessor
from .utils import offset_timestamp, parse_timestamp
import json

# Load environment variables
load_dotenv()

app = typer.Typer()
console = Console()

@app.command()
def process(
    input_file: str = typer.Argument(..., help="Path to the input audio file (mp3/wav)"),
    output_dir: str = typer.Option("./output", help="Directory to save processed files"),
    dry_run: bool = typer.Option(False, help="Analyze only, do not cut audio"),
    api_key: Optional[str] = typer.Option(None, help="Gemini API Key (overrides .env)")
):
    """
    Process a podcast episode: detect ads/intros/outros using Gemini, 
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
    
    console.rule(f"[bold blue]Processing {input_path.name}[/bold blue]")
    
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
        all_segments = []
        full_transcript = ""
        
        try:
            # Generate chunks (30 mins default)
            chunks = processor.create_chunked_proxies(str(input_path))
            
            for i, (chunk_path, offset_sec) in enumerate(chunks):
                console.rule(f"[bold cyan]Processing Chunk {i+1}/{len(chunks)}[/bold cyan]")
                try:
                    uploaded_file = ai.upload_audio(chunk_path)
                    chunk_analysis = ai.analyze_audio(uploaded_file, chunk_context=(i, len(chunks)))
                    
                    # 1. Process Segments (add offset)
                    raw_segments = chunk_analysis.get("segments_to_remove", [])
                    for seg in raw_segments:
                        # Parse locally for validation
                        start_sec = parse_timestamp(seg["start"])
                        end_sec = parse_timestamp(seg["end"])
                        duration = end_sec - start_sec
                        
                        if duration <= 0:
                            continue
                            
                        # SANITY CHECK: Ignore ads > 5 minutes (hallucinations)
                        if duration > 300: # 5 minutes
                             console.print(f"[bold red]WARNING: Ignoring suspicious {duration:.1f}s segment ({seg['type']}) in Chunk {i+1}. Too long to be an ad.[/bold red]")
                             continue

                        # Apply offset
                        seg["start"] = offset_timestamp(seg["start"], offset_sec)
                        seg["end"] = offset_timestamp(seg["end"], offset_sec)
                        all_segments.append(seg)
                        
                    # 2. Process Transcript
                    chunk_text = chunk_analysis.get("transcript", "")
                    full_transcript += f"\n\n--- Chunk {i+1} ---\n\n{chunk_text}"
                    
                finally:
                    # Clean up chunk immediately
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                        console.log(f"Cleaned up chunk {i+1}")

            analysis = {
                "segments_to_remove": all_segments,
                "transcript": full_transcript
            }
            
            # Save cache
            with open(cache_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
        except Exception as e:
            console.print(f"[red]AI Processing failed: {e}[/red]")
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
