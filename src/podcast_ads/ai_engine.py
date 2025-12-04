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
        # Pro model provides better semantic reasoning for ad detection
        self.model_name = "gemini-2.5-flash-lite"
        self.model = genai.GenerativeModel(self.model_name)

    def analyze_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """
        Sends the raw Whisper JSON to Gemini to find semantic ad boundaries.
        """
        console.log(f"[cyan]Reading transcript from {transcript_path}...[/cyan]")
        with open(transcript_path, 'r') as f:
            # We pass the raw JSON string to the model so it sees timestamps + text
            raw_data = f.read()

        prompt = """
        You are an expert Podcast Editor. 
        I am providing you with a raw JSON transcript of an episode. Each segment has a `start`, `end`, and `text`.
        
        **Your Goal:** Identify and remove non-content segments based on the text semantic cues.
        
        **Look for these semantic cues:**
        *   **Intro:** Host introductions, welcome messages, 'Welcome to the show', theme music lyrics.
        *   **Ads:** Phrases like 'Sponsored by', 'Use code', 'Go to [website]', 'Brought to you by', or sudden topic shifts to products/services.
        *   **Outro:** 'Thanks for listening', 'Rate and review', 'See you next week'.
        
        **Output 1: Cut List**
        Return a JSON list of time ranges to REMOVE (`segments_to_remove`). 
        - Use the exact `start` timestamp of the first sentence of the ad.
        - Use the exact `end` timestamp of the last sentence of the ad.
        - format timestamps as HH:MM:SS.mmm
        
        **Output 2: Clean Transcript**
        Provide a cleaned, readable version of the content. 
        - Remove the ads/intros/outros text.
        - Fix capitalization and punctuation.
        - Group short segments into proper paragraphs with Speaker labels if possible.
        
        Return the result STRICTLY as a JSON object:
        {
            "segments_to_remove": [
                {"type": "intro", "start": "HH:MM:SS", "end": "HH:MM:SS"},
                {"type": "ad", "start": "HH:MM:SS", "end": "HH:MM:SS"}
            ],
            "transcript": "The cleaned text..."
        }
        """
        
        console.log("[cyan]Sending transcript to Gemini for analysis...[/cyan]")
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            full_text = ""
            try:
                # Non-streaming is safer for massive context
                with console.status(f"[bold cyan]Gemini is analyzing text (Attempt {attempt+1})...[/bold cyan]"):
                    response = self.model.generate_content(
                        [prompt, raw_data], # Pass the huge JSON string as user content
                        generation_config={
                            "response_mime_type": "application/json",
                            "max_output_tokens": 8192
                        },
                        safety_settings=safety_settings,
                        stream=False
                    )
                    full_text = response.text
                
                console.log(f"[green]Analysis complete. Response size: {len(full_text)} chars.[/green]")
                
                # Clean response
                text = full_text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    console.print("[yellow]Warning: Response JSON was malformed. Attempting manual recovery...[/yellow]")
                    # Fallback extraction logic
                    segments = []
                    transcript = ""
                    
                    seg_match = re.search(r'"segments_to_remove":\s*(\[.*?\])', text, re.DOTALL)
                    if seg_match:
                        try:
                            segments = json.loads(seg_match.group(1))
                        except:
                            pass
                    
                    trans_match = re.search(r'"transcript":\s*"(.*)', text, re.DOTALL)
                    if trans_match:
                        transcript = trans_match.group(1).rstrip('"}')
                    
                    return {"segments_to_remove": segments, "transcript": transcript}

            except Exception as e:
                last_exception = e
                console.print(f"[yellow]API Error (Attempt {attempt+1}): {repr(e)}[/yellow]")
                time.sleep(5)

        raise last_exception or RuntimeError("Failed to analyze transcript")
