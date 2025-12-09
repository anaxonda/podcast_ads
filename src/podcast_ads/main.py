import os
import typer
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
import shutil # For copying files
import yt_dlp # For MediaDownloader's internal info extraction
import ffmpeg # For direct ffmpeg calls in main
import hashlib # For stable hashing of generic URLs

from .ai_engine import AIEngine
from .processor import AudioProcessor
from .media_downloader import MediaDownloader
from .player import Player
from .utils import parse_timestamp 

import json
import re # For cleaning filenames
from urllib.parse import urlparse # For generic URL parsing

# Load environment variables
load_dotenv()

app = typer.Typer()
console = Console()

IS_ANDROID = "com.termux" in os.environ.get("PREFIX", "")

# --- File Stem Utilities ---
def _sanitize_file_stem(value: str, max_len: int = 80) -> str:
    """Sanitize and trim a filename stem."""
    sanitized = re.sub(r'[^\w\-_\.]', '', value)
    sanitized = sanitized.strip("._-")
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized or "media_item"

def _short_hash(value: str, length: int = 10) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:length]

def _extract_youtube_id(parsed_url) -> Optional[str]:
    """
    Best-effort YouTube ID extraction without network calls.
    Supports typical watch URLs and youtu.be shortlinks.
    """
    if not parsed_url:
        return None

    query = parsed_url.query or ""
    if "v=" in query:
        for part in query.split("&"):
            if part.startswith("v="):
                vid = part.replace("v=", "").strip()
                if vid:
                    return vid

    path_parts = (parsed_url.path or "").strip("/").split("/")
    if parsed_url.netloc and "youtu.be" in parsed_url.netloc and path_parts:
        return path_parts[-1] or None

    # Fallback: last path segment if it looks like an ID
    if path_parts:
        candidate = path_parts[-1]
        if len(candidate) in (11, 12): # common YT id lengths
            return candidate
    return None

def _build_candidate_stems(current_input_target: str, is_url: bool, is_youtube: bool) -> List[str]:
    """
    Returns ordered candidate stems to look for cached artifacts.
    1) Stable slug based on normalized input (primary)
    2) Legacy/local stem fallbacks (to reuse previous outputs)
    """
    candidates: List[str] = []
    normalized_target = current_input_target.strip()

    if is_url:
        parsed = urlparse(normalized_target)
        netloc = (parsed.netloc or "").lower()
        path_stem = Path(parsed.path).stem
        short = _short_hash(normalized_target, length=8)

        if is_youtube:
            vid = _extract_youtube_id(parsed)
            base_label = vid or path_stem or "youtube"
            primary = _sanitize_file_stem(f"{base_label}_{short}")
            candidates.append(primary)
            # Legacy: bare video id or title-based names (title handled later on cache miss)
            if vid:
                candidates.append(_sanitize_file_stem(vid))
        else:
            domain = netloc.split(":")[0] if netloc else "media"
            slug_base = path_stem if len(path_stem) >= 3 else "audio"
            primary = _sanitize_file_stem(f"{domain}_{slug_base}_{short}")
            candidates.append(primary)
    else:
        abs_path = str(Path(current_input_target).expanduser().resolve())
        base_stem = Path(current_input_target).stem
        primary = _sanitize_file_stem(f"{base_stem}_{_short_hash(abs_path, length=8)}")
        candidates.append(primary)
        # Legacy/local fallback: plain stem (old behavior)
        candidates.append(_sanitize_file_stem(base_stem))

    # Remove dups while preserving order
    deduped = []
    seen = set()
    for stem in candidates:
        if stem not in seen:
            deduped.append(stem)
            seen.add(stem)
    return deduped

# --- Helper Functions for Output Generation ---
def _generate_lua_script(file_stem: str, out_path: Path, segments_to_remove: List[Dict]) -> str:
    """Generates and saves an MPV Lua script for skipping segments."""
    lua_skips = "local skips = {\n"
    for seg in segments_to_remove:
        start = seg.get('start')
        end = seg.get('end')
        if start is None or end is None: continue
        
        start = float(start)
        end = float(end)
        lua_skips += f"    {{ start = {start}, stop = {end} }},\n"
    lua_skips += "}\n"
    
    lua_script_content = lua_skips + """
mp.add_periodic_timer(0.25, function()
    local pos = mp.get_property_number("time-pos")
    if not pos then return end
    
    for i, skip in ipairs(skips) do
        if pos >= skip.start and pos < skip.stop then
            mp.set_property_number("time-pos", skip.stop)
            mp.osd_message("Auto-Skipped Ad Section")
            break -- Only skip one segment at a time
        end
    end
end)
"""
    script_path = out_path / f"{file_stem}_skips.lua"
    with open(script_path, "w") as f:
        f.write(lua_script_content)
    console.log(f"[green]Generated MPV skip script: {script_path.name}[/green]")
    return str(script_path)

def _generate_srt_file(file_stem: str, out_path: Path, transcript_json_path: str, segments_to_remove: List[Dict]) -> str:
    """Generates and saves an SRT subtitle file from Whisper JSON, filtering out ads."""
    if not transcript_json_path or not os.path.exists(transcript_json_path):
        console.print("[yellow]No Whisper transcript found. Cannot generate SRT.[/yellow]")
        return None

    with open(transcript_json_path, 'r') as f:
        whisper_data = json.load(f)
    
    srt_content = ""
    seq = 1
    for w_seg in whisper_data.get("segments", []):
        start_sec = w_seg.get("start", 0)
        end_sec = w_seg.get("end", 0)
        text = w_seg.get("text", "").strip()
        
        if not text: continue

        # Filter Ads (Overlap Check)
        midpoint = (start_sec + end_sec) / 2
        is_ad = False
        for r_seg in segments_to_remove:
            r_start = r_seg.get('start')
            r_end = r_seg.get('end')
            if r_start is None or r_end is None: continue
            
            if midpoint >= float(r_start) and midpoint <= float(r_end):
                is_ad = True
                break
        
        if is_ad: continue

        # Format time
        start_srt = f"{int(start_sec // 3600):02d}:{int((start_sec % 3600) // 60):02d}:{int(start_sec % 60):02d},{int((start_sec % 1) * 1000):03d}"
        end_srt = f"{int(end_sec // 3600):02d}:{int((end_sec % 3600) // 60):02d}:{int(end_sec % 60):02d},{int((end_sec % 1) * 1000):03d}"

        srt_content += f"{seq}\n{start_srt} --> {end_srt}\n{text}\n\n"
        seq += 1
            
    srt_path = out_path / f"{file_stem}_clean.srt"
    with open(srt_path, "w") as f:
        f.write(srt_content)
    console.log(f"[green]Generated clean SRT: {srt_path.name}[/green]")
    return str(srt_path)

# --- Main CLI Command ---
@app.command()
def process(
    input_path_str: str = typer.Argument(..., help="Path to input file, directory, or media URL", metavar="INPUT"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save processed files"),
    model_size: str = typer.Option("tiny", help="Whisper model size (tiny, small, medium)"),
    api_key: Optional[str] = typer.Option(None, help="Gemini API Key (overrides .env)"),
    
    # --- Action Flags ---
    play: bool = typer.Option(False, "--play", help="Play the media with ads skipped (video stream)"),
    play_audio: bool = typer.Option(False, "--play-audio", help="Play the media with ads skipped (audio stream)"),
    save_clean: bool = typer.Option(False, "--save-clean", help="Save a cut/stitched clean media file (original format)"),
    save_clean_audio: bool = typer.Option(False, "--save-clean-audio", help="Save a cut/stitched clean audio file (MP3)"),
    save_transcript: bool = typer.Option(False, "--save-transcript", help="Save the cleaned Markdown transcript"),
    save_subs: bool = typer.Option(False, "--save-subs", help="Save the cleaned SRT subtitle file"),
    dry_run: bool = typer.Option(False, help="Analyze only, do not perform cuts or saves (except analysis.json and skips.lua)"),
):
    """
    Process media to detect ads/intros/outros.
    By default, generates _analysis.json and _skips.lua. Use flags for other actions.
    """
    
    # 1. Setup
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        console.print("[red]Error: GEMINI_API_KEY not found in .env or arguments.[/red]")
        raise typer.Exit(code=1)
        
    # Determine Output Directory
    if output_dir is None:
        if IS_ANDROID:
            output_dir = "/storage/emulated/0/Download/PodcastAds"
        else:
            output_dir = "./output"
            
    out_path = Path(output_dir)
    
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if IS_ANDROID:
            console.print("[red]Permission denied creating output dir.[/red]")
            console.print("[yellow]Run 'termux-setup-storage' to allow access to Downloads.[/yellow]")
            console.print("Falling back to local ./output folder.")
            out_path = Path("./output")
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            raise

    console.print(f"[dim]Output directory: {out_path}[/dim]")
    
    # 2. Discover Files
    files_to_process = []
    is_url_input = input_path_str.startswith("http")
    
    if is_url_input:
        files_to_process = [input_path_str]
        console.print(f"[green]Processing URL: {input_path_str}[/green]")
    else:
        input_path = Path(input_path_str)
        if not input_path.exists():
            console.print(f"[red]Error: Input {input_path} not found.[/red]")
            raise typer.Exit(code=1)
            
        if input_path.is_dir():
            extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.opus', '*.ogg']
            for ext in extensions:
                files_to_process.extend([str(p) for p in sorted(input_path.rglob(ext))])
            # Remove duplicates
            files_to_process = sorted(list(set(files_to_process)))
            console.print(f"[green]Found {len(files_to_process)} audio files in directory.[/green]")
        else:
            files_to_process = [str(input_path)]

    # 3. Initialize Engines
    ai = AIEngine(key)
    processor = AudioProcessor()

    # 4. Process Loop
    success_count = 0
    fail_count = 0

    for current_input_target in files_to_process:
        # Determine if current item is a YouTube URL
        is_youtube = "youtube.com" in current_input_target or "youtu.be" in current_input_target
        name_label = current_input_target if is_url_input else Path(current_input_target).name
        console.rule(f"[bold blue]Processing {name_label} ({files_to_process.index(current_input_target)+1}/{len(files_to_process)})[/bold blue]")
        
        try:
            _process_single_item_logic(
                current_input_target=current_input_target,
                out_path=out_path,
                ai=ai,
                processor=processor,
                model_size=model_size,
                is_youtube=is_youtube, # Pass YouTube specific flag
                # --- Action Flags ---
                play=play,
                play_audio=play_audio,
                save_clean=save_clean,
                save_clean_audio=save_clean_audio,
                save_transcript=save_transcript,
                save_subs=save_subs,
                dry_run=dry_run
            )
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]Failed to process {name_label}: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

    console.rule("[bold green]Batch Complete[/bold green]")
    console.print(f"Processed: {success_count} | Failed: {fail_count}")

# --- Core Logic for Single Item Processing ---
def _process_single_item_logic(
    current_input_target: str,
    out_path: Path,
    ai: AIEngine,
    processor: AudioProcessor,
    model_size: str,
    is_youtube: bool, # New flag to differentiate YouTube for SB
    # --- Action Flags ---
    play: bool,
    play_audio: bool,
    save_clean: bool,
    save_clean_audio: bool,
    save_transcript: bool,
    save_subs: bool,
    dry_run: bool
):
    is_url = current_input_target.startswith("http") # True if any HTTP URL
    normalized_input = current_input_target.strip()
    segments_to_remove: List[Dict] = []
    
    transcript_json_path: Optional[str] = None # Path to Whisper/YT captions JSON
    actual_media_path: Optional[Path] = None # Path to local media file (downloaded or original)
    
    # --- Determine File Stem Candidates (stable + legacy) ---
    file_stem_candidates = _build_candidate_stems(normalized_input, is_url=is_url, is_youtube=is_youtube)
    file_stem = file_stem_candidates[0] if file_stem_candidates else "generic_media_item"

    # Keep original path handy for local inputs
    if not is_url:
        actual_media_path = Path(normalized_input)
    
    # Legacy YouTube stem (title-based) for cache reuse; only compute if needed
    legacy_title_stem: Optional[str] = None

    def _try_legacy_ytdlp_stem() -> Optional[str]:
        nonlocal legacy_title_stem
        if legacy_title_stem is not None:
            return legacy_title_stem
        try:
            info = yt_dlp.YoutubeDL({'skip_download': True, 'quiet': True, 'no_warnings': True}).extract_info(normalized_input, download=False)
            legacy_title_stem = _sanitize_file_stem(info.get('title') or info.get('id') or "")
            return legacy_title_stem or None
        except Exception:
            legacy_title_stem = None
            return None

    # --- Cache Check ---
    cache_file = out_path / f"{file_stem}_analysis.json"
    analysis = None
    
    # --- Cache Discovery (primary + legacy stems) ---
    cache_candidates = [out_path / f"{stem}_analysis.json" for stem in file_stem_candidates]

    def _load_first_existing(candidates: List[Path]) -> Optional[Path]:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    found_cache = _load_first_existing(cache_candidates)

    # Last-chance legacy: title-based youtube stem (only if nothing matched)
    if not found_cache and is_url and is_youtube:
        legacy_title = _try_legacy_ytdlp_stem()
        if legacy_title:
            legacy_path = out_path / f"{legacy_title}_analysis.json"
            file_stem_candidates.append(legacy_title)
            cache_candidates.append(legacy_path)
            found_cache = _load_first_existing([legacy_path])

    if found_cache:
        file_stem = found_cache.name.replace("_analysis.json", "")
        console.print(f"[yellow]Found cached analysis at {found_cache}. Using it.[/yellow]")
        try:
            with open(found_cache, 'r') as f:
                analysis = json.load(f)
        except Exception:
            console.print("[red]Cache invalid, re-running AI...[/red]")
            analysis = None
    # Refresh cache_file to align with the resolved stem (primary or legacy)
    cache_file = out_path / f"{file_stem}_analysis.json"

    # --- Run Analysis if not in Cache ---
    if not analysis:
        # Initialize MediaDownloader if URL
        md_loader: Optional[MediaDownloader] = None
        if is_url:
            md_loader = MediaDownloader(output_dir=str(out_path))

        # STEP 1: Acquire Raw Transcript Segments (SponsorBlock bypasses this for AI)
        if is_url and md_loader:
            # A. SponsorBlock Check (Only for YouTube)
            if is_youtube:
                sb_segments = md_loader.get_sponsorblock_segments(current_input_target)
                if sb_segments:
                    segments_to_remove = sb_segments
                    console.print("[green]Using SponsorBlock segments. Skipping AI analysis.[/green]")
            
            if not segments_to_remove: # If no SB segments or not YouTube, proceed to AI
                # B. Gemini Fallback: Try to Download Captions (Only for YouTube)
                if is_youtube:
                    transcript_json_path = md_loader.download_captions(current_input_target)
                
                # C. Existing Local Transcript (Optimization)
                if not transcript_json_path:
                     potential_transcript_path = out_path / f"{file_stem}.json"
                     if potential_transcript_path.exists():
                         transcript_json_path = str(potential_transcript_path)
                         console.print(f"[green]Found existing local transcript: {transcript_json_path}[/green]")

                # D. Fallback to Audio Download + Whisper
                if not transcript_json_path:
                    console.print("[yellow]No suitable captions found. Downloading audio for Whisper...[/yellow]")
                    # Download audio for Whisper if no captions or not YouTube
                    actual_media_path = Path(md_loader.download_stream(current_input_target, format_mode='audio', custom_filename=file_stem))
        else: # Local file, actual_media_path is already set
            pass # actual_media_path already set from input_path

        # If segments_to_remove already populated by SB, skip AI. Else, run AI pipeline.
        if not segments_to_remove:
            # If still no transcript_json_path, it means it's a local file or YT audio just downloaded
            if not transcript_json_path:
                if not actual_media_path: 
                    raise ValueError("No media path resolved for Whisper transcription.")
                    
                transcript_json_path = processor.transcribe_local(
                    str(actual_media_path), 
                    model_size=model_size,
                    output_dir=str(out_path)
                )
            
            # STEP 2: Semantic Analysis (Gemini)
            analysis_result = ai.analyze_transcript(transcript_json_path)
            segments_to_remove = analysis_result.get("segments_to_remove", [])
            # We ignore 'transcript_segments' from AI now, as we build it locally
            
        # --- Save Analysis to Cache ---
        cache_payload = {
            "input_meta": {
                "input": normalized_input,
                "is_url": is_url,
                "is_youtube": is_youtube,
                "file_stem": file_stem,
                "schema_version": "v1"
            },
            "segments_to_remove": segments_to_remove,
            "transcript_segments": []
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_payload, f, indent=2)
    else:
        # Load segments from cache
        segments_to_remove = analysis.get("segments_to_remove", [])


    console.print("\n[bold]Identified Segments to Remove:[/bold]")
    for seg in segments_to_remove:
        seg_type = seg.get("type", "AD").upper()
        start_t = seg.get('start')
        end_t = seg.get('end')
        console.print(f" - [red]{seg_type}[/red]: {start_t} -> {end_t}")

    # --- STEP 3: Default Outputs (Always Generated) ---
    # Generate and save _skips.lua
    skips_lua_path = _generate_lua_script(file_stem, out_path, segments_to_remove)

    # --- STEP 4: Conditional Actions based on Flags ---
    # --- Playback Actions ---
    if play or play_audio:
        player = Player()
        # For play, use the original input target (URL or local path)
        player.play_with_skips(current_input_target, skips_lua_path, audio_only=play_audio, segments=segments_to_remove)
        return # Play mode finishes the process for this item

    # --- Dry Run ---
    if dry_run:
        console.print("[yellow]Dry run. Skipping media cuts and explicit saves.[/yellow]")
        return # Dry run finishes early, after generating default outputs


    # --- Media Download / Processing for Save Flags ---
    if save_clean or save_clean_audio:
        # Resolve actual_media_path by downloading if it's a URL and not already downloaded
        if is_url and not actual_media_path:
            md_loader = MediaDownloader(output_dir=str(out_path))
            if save_clean_audio:
                actual_media_path = Path(md_loader.download_stream(current_input_target, format_mode='audio', custom_filename=file_stem))
            else: # save_clean (video+audio)
                actual_media_path = Path(md_loader.download_stream(current_input_target, format_mode='video', custom_filename=file_stem))
        
        if not actual_media_path:
            raise ValueError("Media path not resolved for cutting.")

        final_media_output_path: Path
        if save_clean_audio:
            # Always output MP3 for clean audio
            final_media_output_path = out_path / f"{file_stem}_clean.mp3" 
            # If original media is not MP3, convert it before cutting
            if actual_media_path.suffix.lower() not in ['.mp3', '.m4a', '.wav']: # Common audio formats
                console.print(f"[cyan]Converting {actual_media_path.suffix} to MP3 for clean audio output...[/cyan]")
                temp_audio_path = out_path / f"{file_stem}_temp_audio.mp3"
                try:
                    (
                        ffmpeg.input(str(actual_media_path))
                        .output(str(temp_audio_path), acodec='libmp3lame', audio_bitrate='192k')
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    processor.cut_and_merge(str(temp_audio_path), str(final_media_output_path), segments_to_remove)
                    if temp_audio_path.exists():
                        temp_audio_path.unlink()
                except Exception as e:
                    console.print(f"[red]Error converting to MP3 for save_clean_audio: {e}[/red]")
                    raise
            else: # Already an audio file
                processor.cut_and_merge(str(actual_media_path), str(final_media_output_path), segments_to_remove)
        else: # save_clean (original format, or best video+audio for YouTube)
            final_media_output_path = out_path / f"{file_stem}_clean{actual_media_path.suffix}"
            processor.cut_and_merge(str(actual_media_path), str(final_media_output_path), segments_to_remove)
        
        console.print(f"[green]Clean media saved to {final_media_output_path.name}[/green]")

    # --- Text Artifacts (Conditionally Saved) ---
    if save_transcript or save_subs:
        # We need transcript_json_path. If we loaded from cache, it might be None.
        if not transcript_json_path:
             possible_path = out_path / f"{file_stem}.json"
             if possible_path.exists():
                 transcript_json_path = str(possible_path)
        
        if not transcript_json_path:
             console.print("[yellow]No Whisper transcript found. Cannot generate text outputs.[/yellow]")
        else:
            if save_transcript:
                clean_transcript_path = out_path / f"{file_stem}_transcript.md"
                with open(transcript_json_path, 'r') as f:
                    whisper_data = json.load(f)
                
                transcript_text = ""
                for w_seg in whisper_data.get("segments", []):
                    w_text = w_seg.get("text", "").strip()
                    if not w_text: continue
                    
                    midpoint = (w_seg.get("start", 0) + w_seg.get("end", 0)) / 2
                    is_ad = False
                    for r_seg in segments_to_remove:
                        start_val = r_seg.get('start')
                        end_val = r_seg.get('end')
                        if start_val is None or end_val is None: continue
                        if midpoint >= float(start_val) and midpoint <= float(end_val):
                            is_ad = True
                            break
                    
                    if not is_ad:
                        transcript_text += f"{w_text}\n"
                
                with open(clean_transcript_path, "w") as f:
                    f.write(f"# Transcript: {file_stem}\n\n")
                    f.write(transcript_text)
                console.print(f"[green]Clean transcript saved to {clean_transcript_path.name}[/green]")

            if save_subs:
                _generate_srt_file(file_stem, out_path, transcript_json_path, segments_to_remove)
