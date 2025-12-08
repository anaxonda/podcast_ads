import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console

console = Console()
IS_ANDROID = "com.termux" in os.environ.get("PREFIX", "")

class Player:
    def __init__(self):
        pass

    def play_with_skips(self, media_path: str, script_path: str, audio_only: bool = False, segments: Optional[List[Dict]] = None):
        """
        Plays the media (URL or file) using MPV.
        
        Args:
            media_path: URL or file path to play.
            script_path: Path to the generated Lua script (for PC).
            audio_only: Whether to disable video.
            segments: Raw list of segments (Required for Android generation).
        """
        if IS_ANDROID:
            self._play_android(media_path, segments)
        else:
            self._play_pc(media_path, script_path, audio_only)

    def _play_pc(self, media_path: str, script_path: str, audio_only: bool):
        console.print(f"[green]Launching MPV player...[/green]")
        console.print(f"[dim]Media: {media_path}[/dim]")
        
        cmd = [
            "mpv",
            media_path,
            f"--script={Path(script_path).absolute()}",
            "--force-window=immediate"
        ]
        
        if audio_only:
            cmd.append("--no-video")
            console.print("[cyan]Audio-only mode enabled.[/cyan]")
        
        try:
            subprocess.run(cmd)
        except FileNotFoundError:
            console.print("[red]Error: 'mpv' player not found. Please install mpv.[/red]")

    def _play_android(self, media_path: str, segments: List[Dict]):
        if not segments:
            console.print("[yellow]No segments provided for Android playback generation.[/yellow]")
            return

        # Android Target: mpvKt config dir (Requires user setup)
        # We write a temporary script that ONLY applies to the current filename
        target_dir = Path("/storage/emulated/0/Videos/mpv_config/scripts")
        
        if not target_dir.exists():
            console.print(f"[red]Error: mpvKt scripts directory not found: {target_dir}[/red]")
            console.print("[yellow]Please create '/sdcard/Videos/mpv_config/scripts' and configure mpvKt to use it.[/yellow]")
            return

        # Extract filename for Guard Clause
        # If URL, match the full string? Or just be loose?
        # Let's match the filename end.
        if "://" in media_path:
            target_name = media_path # Match full URL
        else:
            target_name = Path(media_path).name

        # Generate Android-specific Lua
        lua_content = "local skips = {\n"
        for seg in segments:
            start = float(seg.get('start', 0))
            end = float(seg.get('end', 0))
            lua_content += f"    {{ start = {start}, stop = {end} }},\n"
        lua_content += "}\n"
        
        lua_content += f"""
local target_media = "{target_name}"

mp.add_periodic_timer(0.25, function()
    -- Guard: Check if current media matches our target
    local path = mp.get_property("path")
    if not path then return end
    
    -- Simple string check (contains)
    if not string.find(path, target_media, 1, true) then 
        return 
    end

    local pos = mp.get_property_number("time-pos")
    if not pos then return end
    
    for i, skip in ipairs(skips) do
        if pos >= skip.start and pos < skip.stop then
            mp.set_property_number("time-pos", skip.stop)
            mp.osd_message("Auto-Skipped Ad")
            break 
        end
    end
end)
"""
        # Use a unique name based on the target to allow persistent storage
        # Sanitize target_name for filesystem (keep alphanumeric, dots, dashes, underscores)
        safe_name = "".join(c for c in target_name if c.isalnum() or c in "._-")
        script_file = target_dir / f"{safe_name}_skips.lua"
        
        with open(script_file, "w") as f:
            f.write(lua_content)
            
        console.print(f"[green]Wrote persistent skip script to {script_file}[/green]")
        # console.print(f"[dim]This script will auto-load in mpvKt for: {target_name}[/dim]")

        # Launch mpvKt
        console.print(f"[green]Launching mpvKt...[/green]")
        cmd = [
            "am", "start",
            "-a", "android.intent.action.VIEW",
            "-d", media_path,
            "-n", "live.mehiz.mpvkt/is.xyz.mpv.MPVActivity"
        ]
        subprocess.run(cmd)
