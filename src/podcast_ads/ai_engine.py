import os
from typing import List, Dict, Any
import json
import time
import re
from rich.console import Console
from openai import OpenAI

# Optional Google Import
try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

console = Console()

class AIEngine:
    def __init__(self, api_key: str = None):
        # API Keys
        self.google_api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.or_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Model Chain Configuration
        env_order = os.getenv("AI_MODEL_ORDER")
        if env_order:
            self.model_chain = [m.strip() for m in env_order.split(",") if m.strip()]
        else:
            # Default fallback chain
            self.model_chain = [
                "gemini-pro-latest", 
                "gemini-2.5-flash", 
                "openrouter/x-ai/grok-4.1-fast"
            ]
            
        # Initialize Clients
        if HAS_GOOGLE and self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            
        self.or_client = None
        if self.or_api_key:
            self.or_client = OpenAI(
                base_url=self.openai_base_url,
                api_key=self.or_api_key,
            )

    def analyze_transcript(self, transcript_path: str) -> Dict[str, Any]:
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
            
            # Unified execution loop
            response_data = self._process_chunk_with_fallbacks(prompt, chunk_payload, i+1)
            
            if response_data:
                all_remove_segments.extend(response_data.get("segments_to_remove", []))

        return {"segments_to_remove": all_remove_segments}

    def _process_chunk_with_fallbacks(self, prompt, chunk_data, chunk_num) -> Dict:
        """Iterates through the model chain until one succeeds."""
        
        for model_id in self.model_chain:
            is_openrouter = model_id.startswith("openrouter/") or model_id.startswith("or/")
            clean_model_name = model_id.replace("openrouter/", "").replace("or/", "")
            
            try:
                if is_openrouter:
                    return self._call_openrouter(clean_model_name, prompt, chunk_data, chunk_num)
                else:
                    return self._call_gemini(clean_model_name, prompt, chunk_data, chunk_num)
            except Exception as e:
                # If specific provider failed, log and continue to next model
                console.print(f"[yellow]Model {clean_model_name} failed: {e}. Trying next...[/yellow]")
                continue
        
        console.print(f"[bold red]All models in chain failed for Chunk {chunk_num}.[/bold red]")
        return {}

    def _call_gemini(self, model_name, prompt, chunk_data, chunk_num) -> Dict:
        if not HAS_GOOGLE or not self.google_api_key:
            raise RuntimeError("Google API not configured")

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        with console.status(f"[bold cyan]Analyzing Chunk {chunk_num} (Google: {model_name})...[/bold cyan]"):
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [prompt, chunk_data],
                generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192},
                safety_settings=safety_settings,
                stream=False
            )
            text = response.text.strip()
            return self._parse_response(text)

    def _call_openrouter(self, model_name, prompt, chunk_data, chunk_num) -> Dict:
        if not self.or_client:
            raise RuntimeError("OpenRouter API not configured")
            
        with console.status(f"[bold cyan]Analyzing Chunk {chunk_num} (OpenRouter: {model_name})...[/bold cyan]"):
            completion = self.or_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                    {"role": "user", "content": prompt + "\n\nDATA:\n" + chunk_data}
                ],
            )
            text = completion.choices[0].message.content
            return self._parse_response(text)

    def _parse_response(self, text) -> Dict:
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return {"segments_to_remove": data}
            return data
        except json.JSONDecodeError:
            segments = []
            seg_match = re.search(r'"segments_to_remove"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if seg_match:
                try:
                    segments = json.loads(seg_match.group(1))
                except:
                    pass
            return {"segments_to_remove": segments}
