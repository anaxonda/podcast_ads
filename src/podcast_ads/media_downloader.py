import yt_dlp
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console

console = Console()

class MediaDownloader:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_sponsorblock_segments(self, url: str) -> List[Dict[str, Any]]:
        """
        Fetches SponsorBlock segments using yt-dlp CLI.
        Returns list of dicts: {'start': float, 'end': float, 'type': str}
        """
        console.log("[cyan]Checking SponsorBlock database...[/cyan]")
        cmd = [
            "yt-dlp", 
            "--dump-json", 
            "--sponsorblock-mark", "all",
            "--skip-download",
            url
        ]
        try:
            # Run quiet to avoid polluting stdout, capture json
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return []
                
            # yt-dlp might output debug lines before JSON, look for the JSON blob
            output = result.stdout
            # Simple heuristic: parsing the last line usually works for dump-json
            # or just parsing the whole thing if it's clean.
            data = json.loads(output)
            
            chapters = data.get('sponsorblock_chapters', [])
            if not chapters:
                console.log("[dim]No SponsorBlock segments found.[/dim]")
                return []
                
            segments = []
            for chap in chapters:
                segments.append({
                    "start": chap['start_time'],
                    "end": chap['end_time'],
                    "type": chap['title'] # e.g. "Sponsor"
                })
            
            console.print(f"[green]Found {len(segments)} segments via SponsorBlock![/green]")
            return segments
            
        except Exception as e:
            console.log(f"[dim]SponsorBlock check failed or empty: {e}[/dim]")
            return []

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Fetches metadata, subtitles, and sponsorblock info without downloading video.
        """
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,  # Prefer auto-subs if no manual
            'writesub': True,
            'subtitleslangs': ['en'],
            'outtmpl': str(self.output_dir / '%(id)s'),
            'quiet': True,
            'no_warnings': True,
        }

        console.log(f"[cyan]Fetching metadata for {url}...[/cyan]")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info

    def download_captions(self, url: str) -> str:
        """
        Downloads captions and converts them to the JSON format our AIEngine expects.
        Returns path to the JSON transcript.
        """
        # We use write_auto_sub=True to ensure we get something
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'writesub': True,
            'subtitleslangs': ['en.*', 'en'],
            'subtitlesformat': 'json3', # Internal JSON format is easiest to parse
            'outtmpl': str(self.output_dir / '%(id)s'),
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True) # download=True needed to save subs
            video_id = info['id']
            
            # Find the file
            # yt-dlp might save as .en.json3 or .en.vtt depending on availability
            # We look for the json3 file first
            json_path = self.output_dir / f"{video_id}.en.json3"
            
            # If manual subs exist, it might be just .json3? 
            # Let's look for any json3 file with the ID
            found = list(self.output_dir.glob(f"{video_id}*.json3"))
            
            if not found:
                # Fallback: Try VTT and convert? 
                # Ideally json3 is best. Let's trust yt-dlp found something.
                # If not, maybe no subs available.
                console.print("[yellow]No subtitles found by yt-dlp.[/yellow]")
                return None
                
            raw_sub_path = found[0]
            return self._convert_ytdlp_json_to_whisper_json(raw_sub_path)

    def _convert_ytdlp_json_to_whisper_json(self, ytdlp_path: Path) -> str:
        """
        Converts yt-dlp's 'json3' format to the structure our AIEngine expects:
        { "segments": [ {"start": 0.0, "end": 1.0, "text": "..."} ] }
        """
        with open(ytdlp_path, 'r') as f:
            data = json.load(f)
            
        whisper_segments = []
        events = data.get('events', [])
        
        for event in events:
            # yt-dlp json3 format: { "tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "text"}] }
            if 'segs' not in event: continue
            
            text = "".join([s.get('utf8', '') for s in event['segs']]).strip()
            if not text: continue
            
            start_sec = event.get('tStartMs', 0) / 1000.0
            duration_sec = event.get('dDurationMs', 0) / 1000.0
            end_sec = start_sec + duration_sec
            
            whisper_segments.append({
                "start": start_sec,
                "end": end_sec,
                "text": text
            })
            
        output_path = ytdlp_path.with_suffix('.converted.json')
        with open(output_path, 'w') as f:
            json.dump({"segments": whisper_segments}, f, indent=2)
            
        console.log(f"[green]Converted captions to {output_path.name}[/green]")
        return str(output_path)

    def download_stream(self, url: str, format_mode: str = "audio", custom_filename: str = None) -> str:
        """
        Downloads the actual media file.
        format_mode: 'audio' (m4a/mp3) or 'video' (mp4)
        custom_filename: optional stable filename (without extension) to force output name
        """
        console.log(f"[cyan]Downloading {format_mode} stream...[/cyan]")
        
        template = str(self.output_dir / '%(title)s.%(ext)s')
        if custom_filename:
            template = str(self.output_dir / f"{custom_filename}.%(ext)s")

        ydl_opts = {
            'outtmpl': template,
            'quiet': False,
        }
        
        if format_mode == 'audio':
            ydl_opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            })
        else:
            ydl_opts.update({
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            })

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if format_mode == 'audio':
                # post-processor changes ext to mp3
                # We need to predict the final name based on our template or returned filename
                pre, _ = os.path.splitext(filename)
                filename = pre + ".mp3"
            return filename
