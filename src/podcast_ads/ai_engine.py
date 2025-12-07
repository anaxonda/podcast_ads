import os
import google.generativeai as genai
from typing import List, Dict, Any
import json
import time
import re
from rich.console import Console
from openai import OpenAI

console = Console()

class AIEngine:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Fallback chain: Pro (Quality) -> 2.5 Flash (Speed/Quality) -> OpenRouter
        self.models_to_try = ["gemini-pro-latest", "gemini-2.5-flash"]
        self.current_model_idx = 0
        self.model_name = self.models_to_try[0]
        self.model = genai.GenerativeModel(self.model_name)
        
        self.or_api_key = os.getenv("OPENROUTER_API_KEY")
        self.or_client = None
        if self.or_api_key:
            self.or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.or_api_key,
            )

    def analyze_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Sends the raw Whisper JSON to Gemini in chunks to find semantic ad boundaries.
        """
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
                is_quota = "429" in error_str or "resourceexhausted" in error_str or "quota" in error_str or "limit" in error_str
                # Check for Safety Block (Finish Reason 2)
                is_safety = "finishreason" in error_str and "2" in error_str
                
                if is_quota or is_safety:
                    reason = "Quota limit hit" if is_quota else "Safety block"
                    console.print(f"[yellow]{reason} for {self.model_name}. Switching fallback...[/yellow]")
                    
                    self.current_model_idx += 1
                    if self.current_model_idx < len(self.models_to_try):
                        self.model_name = self.models_to_try[self.current_model_idx]
                        self.model = genai.GenerativeModel(self.model_name)
                        console.print(f"[green]Switched to {self.model_name}[/green]")
                        time.sleep(2)
                        continue
                    else:
                        console.print("[red]All Gemini models exhausted.[/red]")
                        return self._call_openrouter_chunk(prompt, chunk_data, chunk_num)

                console.print(f"[yellow]Error Chunk {chunk_num}: {repr(e)}[/yellow]")
                if response and hasattr(response, 'prompt_feedback'):
                    console.print(f"[red]Prompt Feedback: {response.prompt_feedback}[/red]")
                time.sleep(2)
                
        # If loop finishes without success/return, try OpenRouter as last resort
        return self._call_openrouter_chunk(prompt, chunk_data, chunk_num)

    def _call_openrouter_chunk(self, prompt, chunk_data, chunk_num) -> Dict:
        if not self.or_client:
            console.print("[red]No OpenRouter API key found. Skipping chunk.[/red]")
            return {}
            
        console.print(f"[cyan]Fallback to OpenRouter (Chunk {chunk_num})...[/cyan]")
        # Fallback model requested by user
        model = "x-ai/grok-4.1-fast"
        
        try:
            completion = self.or_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                    {"role": "user", "content": prompt + "\n\nDATA:\n" + chunk_data}
                ],
                # response_format={"type": "json_object"} # Not all OR models support this
            )
            text = completion.choices[0].message.content
            
            if text.startswith("```json"): text = text[7:]
            if text.endswith("```"): text = text[:-3]
            
            try:
                data = json.loads(text)
                if isinstance(data, list): return {"segments_to_remove": data}
                return data
            except json.JSONDecodeError:
                # Basic recovery
                seg_match = re.search(r'"segments_to_remove"\s*:\s*(\[.*?\])', text, re.DOTALL)
                if seg_match:
                    try: return {"segments_to_remove": json.loads(seg_match.group(1))}
                    except: pass
                return {}
                
        except Exception as e:
            console.print(f"[red]OpenRouter Failed: {e}[/red]")
            return {}
