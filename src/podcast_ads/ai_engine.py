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
        self.model_name = "gemini-2.5-flash" # Efficient for audio, available in user's list
        self.model = genai.GenerativeModel(self.model_name)

    def upload_audio(self, file_path: str):
        """Uploads audio file to Gemini File API."""
        
        console.log(f"[cyan]Starting upload of {file_path} to Gemini...[/cyan]")
        try:
            audio_file = genai.upload_file(path=file_path)
            console.log(f"[cyan]Upload finished. Waiting for server processing...[/cyan]")
            
            # Wait for processing
            while True:
                # Handle cases where state is Enum (has .name) or String (direct comparison)
                state_obj = getattr(audio_file, 'state', None)
                state_name = state_obj.name if hasattr(state_obj, 'name') else str(state_obj)
                
                if state_name != "PROCESSING":
                    break
                    
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
                
            if state_name == "FAILED":
                raise ValueError("Audio file processing failed on Gemini servers.")
                
            console.log(f"[green]Audio ready for analysis: {audio_file.uri}[/green]")
            return audio_file
        except Exception as e:
            console.print(f"[red]Error uploading file: {repr(e)}[/red]")
            raise

    def analyze_audio(self, audio_file_obj, chunk_context: tuple[int, int] = None) -> Dict[str, Any]:
        """
        Analyzes the audio to find ads, intro, outro and get transcript.
        """
        
        context_instruction = ""
        if chunk_context:
            idx, total = chunk_context
            context_instruction = f"IMPORTANT CONTEXT: This audio file is Chunk {idx+1} of {total} from a longer podcast episode. "
            
            if idx > 0:
                context_instruction += (
                    "The audio begins abruptly in the middle of the timeline. "
                    "- If it starts with conversation, do NOT mark it as an 'Intro'. "
                    "- If it starts in the middle of an Ad, DO mark it as an 'Ad' starting at 00:00:00. "
                )
            
            if idx < total - 1:
                context_instruction += (
                    "The audio ends abruptly and continues in the next chunk. "
                    "- Do NOT mark the end as an 'Outro' unless you hear the final credits/theme of the entire show. "
                    "- If an Ad is playing at the end, mark the 'Ad' ending at the very last second of this file."
                )

        prompt = f"""
        You are a professional podcast editor. Listen to this audio file carefully.
        {context_instruction}
        
        Your tasks are:
        1. Identify the precise start and end timestamps for the **Intro** (music, initial host greeting before content).
        2. Identify any **Mid-roll Ads** (look for 'sponsored by', pitch changes, music beds distinct from content, or explicit ad breaks).
        3. Identify the **Outro** (closing remarks, credits, fading music).
        4. Generate a **cleaned** verbatim transcript of the content. 
           - **EXCLUDE** the text corresponding to the Intro, Ads, and Outro segments you identified.
           - **FORMAT** the text with clear paragraph breaks (\\n\\n) and speaker labels (e.g., '**Host:**').
           - Ensure you capture the conversation flow right up to the cut points so the transcript is readable.
        
        Return the result STRICTLY as a JSON object with this structure (no markdown formatting around the json):
        {{
            "segments_to_remove": [
                {{"type": "intro", "start": "HH:MM:SS", "end": "HH:MM:SS"}},
                {{"type": "ad", "start": "HH:MM:SS", "end": "HH:MM:SS"}},
                {{"type": "outro", "start": "HH:MM:SS", "end": "HH:MM:SS"}}
            ],
            "transcript": "The cleaned transcript with formatting..."
        }}
        
        If you don't find specific segments (like no ads), leave the list empty.
        Ensure timestamps are accurate.
        """

        console.log("[cyan]Sending analysis request to Gemini (this may take a minute for the model to 'listen')...[/cyan]")
        
        # Configure safety settings to avoid blocking podcast content (e.g. crime, news)
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
                # Standard request (Non-streaming) for better stability
                with console.status(f"[bold cyan]Gemini is analyzing (Attempt {attempt+1})...[/bold cyan]"):
                    response = self.model.generate_content(
                        [prompt, audio_file_obj],
                        generation_config={
                            "response_mime_type": "application/json",
                            "max_output_tokens": 8192
                        },
                        safety_settings=safety_settings,
                        stream=False  # CHANGED: Disabled streaming to prevent StopIteration on timeouts
                    )
                    full_text = response.text
                
                console.log(f"[green]Analysis generation complete. Total size: {len(full_text)} chars.[/green]")
                
                # Clean response if model accidentally adds markdown
                text = full_text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    console.print("[yellow]Warning: Response JSON was malformed or truncated. Attempting manual recovery...[/yellow]")
                    
                    # Fallback extraction
                    segments = []
                    transcript = ""
                    
                    # Extract segments list
                    seg_match = re.search(r'"segments_to_remove":\s*(\[.*?\])', text, re.DOTALL)
                    if seg_match:
                        try:
                            segments = json.loads(seg_match.group(1))
                        except:
                            console.print("[red]Could not parse segments list.[/red]")
                    
                    # Extract transcript (greedy match until end or cutoff)
                    trans_match = re.search(r'"transcript":\s*"(.*)', text, re.DOTALL)
                    if trans_match:
                        raw_transcript = trans_match.group(1)
                        # If it ended cleanly with a quote and brace, clean it up
                        if raw_transcript.rstrip().endswith('"}'):
                             transcript = raw_transcript.rstrip()[:-2]
                        elif raw_transcript.rstrip().endswith('"'):
                             transcript = raw_transcript.rstrip()[:-1]
                        else:
                            # Truncated
                            transcript = raw_transcript + "\n\n[TRANSCRIPT TRUNCATED DUE TO LENGTH LIMIT]"
                            
                        # Unescape JSON encoded newlines if needed
                        transcript = transcript.replace('\\n', '\n').replace('\\"', '"')

                    return {
                        "segments_to_remove": segments,
                        "transcript": transcript
                    }
                    
            except Exception as e:
                last_exception = e
                console.print(f"[yellow]Network/API Error (Attempt {attempt+1}/{max_retries}): {repr(e)}[/yellow]")
                if attempt < max_retries - 1:
                    console.print("[yellow]Waiting 5 seconds before retrying...[/yellow]")
                    time.sleep(5)
                else:
                    console.print("[red]Max retries reached.[/red]")
                    
        # If we exit the loop without returning, raise the last error
        if full_text:
             console.print(f"[yellow]Partial response received:[/yellow] {full_text[:500]}...")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Analysis failed without specific exception")
