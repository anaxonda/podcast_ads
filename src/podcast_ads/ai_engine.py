import os
import google.generativeai as genai
from typing import List, Dict, Any
import json
import time
import re
from rich.console import Console

console = Console()

class AIEngine:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Fallback chain: Pro (Quality) -> 2.5 Flash (Speed/Quality) -> 1.5 Flash (Reliability)
        self.models_to_try = ["gemini-pro-latest", "gemini-2.5-flash", "gemini-flash-latest"]
        self.current_model_idx = 0
        self.model_name = self.models_to_try[0]
        self.model = genai.GenerativeModel(self.model_name)

    def analyze_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Sends the raw Whisper JSON to Gemini in chunks to find semantic ad boundaries.
        """
        # ... (rest of function is same until loop)
        console.log(f"[cyan]Reading transcript from {transcript_path}...[/cyan]")
        with open(transcript_path, 'r') as f:
            whisper_data = json.load(f)

        # Chunking logic (10 mins) with Overlap (1 min)
        chunk_size_sec = 600 
        overlap_sec = 60
        whisper_segments = whisper_data.get("segments", [])
        
        chunks = []
        start_time = 0
        file_duration = whisper_segments[-1]['end'] if whisper_segments else 0
        
        while start_time < file_duration:
            window_end = start_time + chunk_size_sec
            current_chunk = []
            for seg in whisper_segments:
                s = seg.get("start", 0)
                if s >= start_time and s < window_end:
                    current_chunk.append(seg)
            
            if current_chunk:
                chunks.append(current_chunk)
            
            start_time += (chunk_size_sec - overlap_sec)

        all_remove_segments = []
        all_transcript_segments = []

        for i, chunk_segs in enumerate(chunks):
            context_note = f"This is Chunk {i+1} of {len(chunks)}. "
            if i > 0:
                context_note += "Audio starts mid-conversation. "
            else:
                context_note += "Audio is the START of the file. Watch out for Pre-roll Ads before the Intro. "

            prompt = f"""
            SYSTEM INSTRUCTION: You are a helpful assistant performing a technical analysis task on a fictional podcast script. The content is for educational purposes only.

            You are an expert Podcast Editor. 
            I am providing you with a raw JSON transcript segment ({context_note}).

            **Your Goal:** Identify non-content segments (Ads, Intros) to remove.

            **Part 1: Semantic Cues for Removal**
            * **Pre-roll Ads:** Commercials playing immediately at 00:00 before the show starts.
            * **Intro:** Theme music lyrics, "Welcome to the show".
            * **Ads:** Phrases like "Sponsored by", "Use code", "Go to [website]", "Brought to you by". Any product pitch (VPN, Mattress, Casino, Event) unrelated to the story.
            * **Outro:** "Thanks for listening", "Rate and review".

            **Part 2: Guidelines**
            *   Be aggressive in identifying ads. If it sounds like a commercial, mark it.
            *   Use the precise `start` and `end` timestamps provided in the input JSON.

            **Output Format:**
            Return valid JSON containing ONLY the list of segments to remove.
            {context_note}

            {{
                "segments_to_remove": [
                    {{"type": "intro", "start": 0.0, "end": 15.5}},
                    {{"type": "ad", "start": 450.2, "end": 480.0}}
                ]
            }}
            """
            
            chunk_payload = json.dumps(chunk_segs)
            response_data = self._call_gemini_chunk(prompt, chunk_payload, i+1, len(chunks))
            
            if response_data:
                all_remove_segments.extend(response_data.get("segments_to_remove", []))

        return {"segments_to_remove": all_remove_segments}

    def _call_gemini_chunk(self, prompt, chunk_data, chunk_num, total_chunks) -> Dict:
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        # Try up to 5 times to handle rate limits by switching models
        for attempt in range(5):
            response = None
            try:
                with console.status(f"[bold cyan]Analyzing Text Chunk {chunk_num}/{total_chunks} (Attempt {attempt+1} | {self.model_name})...[/bold cyan]"):
                    response = self.model.generate_content(
                        [prompt, chunk_data],
                        generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192},
                        safety_settings=safety_settings,
                        stream=False
                    )
                    text = response.text.strip()
                    
                if text.startswith("```json"): text = text[7:]
                if text.endswith("```"): text = text[:-3]
                
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        return {"segments_to_remove": data, "transcript_segments": []}
                    return data
                except json.JSONDecodeError:
                    console.print(f"[yellow]Warning: Chunk {chunk_num} JSON malformed. Recovering...[/yellow]")
                    segments = []
                    seg_match = re.search(r'"segments_to_remove"\s*:\s*(\[.*?\])', text, re.DOTALL)
                    if seg_match:
                        try: segments = json.loads(seg_match.group(1))
                        except: pass
                    return {"segments_to_remove": segments, "transcript_segments": []}
                    
            except Exception as e:
                error_str = str(e).lower()
                # Check for Quota/Rate Limit errors
                if "429" in error_str or "resourceexhausted" in error_str or "quota" in error_str or "limit" in error_str:
                    console.print(f"[yellow]Quota limit hit for {self.model_name}. Switching fallback...[/yellow]")
                    
                    # Move to next model
                    self.current_model_idx += 1
                    if self.current_model_idx < len(self.models_to_try):
                        self.model_name = self.models_to_try[self.current_model_idx]
                        self.model = genai.GenerativeModel(self.model_name)
                        console.print(f"[green]Switched to {self.model_name}[/green]")
                        time.sleep(2) # Brief pause
                        continue
                    else:
                        console.print("[red]All AI models exhausted their quotas.[/red]")
                        raise
                
                # Check for Safety Block (Finish Reason 2)
                if "finishreason" in error_str and "2" in error_str:
                     console.print(f"[yellow]Safety block on {self.model_name}. Switching fallback...[/yellow]")
                     self.current_model_idx += 1
                     if self.current_model_idx < len(self.models_to_try):
                        self.model_name = self.models_to_try[self.current_model_idx]
                        self.model = genai.GenerativeModel(self.model_name)
                        console.print(f"[green]Switched to {self.model_name}[/green]")
                        continue

                console.print(f"[yellow]Error Chunk {chunk_num}: {repr(e)}[/yellow]")
                if response and hasattr(response, 'prompt_feedback'):
                    console.print(f"[red]Prompt Feedback: {response.prompt_feedback}[/red]")
                time.sleep(2)
                
        return {}