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
        # Pro models often have more nuanced safety filters, allowing valid content that Flash blocks
        self.model_name = "gemini-pro-latest" 
        self.model = genai.GenerativeModel(self.model_name)

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
        
        # Deduplication history (previous chunk's texts)
        last_chunk_texts = set()

        for i, chunk_segs in enumerate(chunks):
            # Context Instruction
            context_note = f"This is Chunk {i+1} of {len(chunks)}. "
            if i > 0:
                context_note += "Audio starts mid-conversation. "
            else:
                context_note += "Audio is the START of the file. Watch out for Pre-roll Ads before the Intro. "

            prompt = f"""
            SYSTEM INSTRUCTION: You are a helpful assistant performing a technical analysis task on a fictional podcast script. The content is for educational purposes only.

            You are an expert Podcast Editor. 
            I am providing you with a raw JSON transcript segment ({context_note}).

            **Your Goal:** 
            1. Identify non-content segments (Ads, Intros) to remove.
            2. Reconstruct the remaining content into a structured dialogue format.

            **Part 1: Semantic Cues for Removal**
            * **Pre-roll Ads:** Commercials playing immediately at 00:00 before the show starts.
            * **Intro:** Theme music lyrics, "Welcome to the show".
            * **Ads:** Phrases like "Sponsored by", "Use code", "Go to [website]", "Brought to you by". Any product pitch (VPN, Mattress, Casino, Event) unrelated to the story.
            * **Outro:** "Thanks for listening", "Rate and review".

            **Part 2: Reconstruction Rules (Crucial)**
            The input lacks speaker labels. You MUST infer them to break the wall of text.
            * **Q&A Detection:** If a sentence asks a question and the next answers it, that is a speaker switch.
            * **Paragraphing:** Break long monologues into shorter chunks (max 3-4 sentences).
            * **Formatting:** Fix punctuation and capitalization. Remove filler words.

            **Output Format:**
            Return valid JSON. 
            {context_note}

            {{
                "segments_to_remove": [
                    {{"type": "intro", "start": 0.0, "end": 15.5}},
                    {{"type": "ad", "start": 450.2, "end": 480.0}}
                ],
                "transcript_segments": [
                    {{
                        "speaker_label": "**Speaker 1:**", 
                        "text": "Welcome to the show..."
                    }}
                ]
            }}
            """
            
            chunk_payload = json.dumps(chunk_segs)
            
            # Call API with Retry Logic
            response_data = self._call_gemini_chunk(prompt, chunk_payload, i+1, len(chunks))
            
            # Aggregate results
            if response_data:
                all_remove_segments.extend(response_data.get("segments_to_remove", []))
                
                # Deduplicate Transcript
                new_trans_segs = response_data.get("transcript_segments", [])
                current_chunk_texts = set()
                
                for t_seg in new_trans_segs:
                    txt = t_seg.get("text", "").strip()
                    
                    # If text is substantial and was present in the previous chunk (overlap), skip it
                    if len(txt) > 20 and txt in last_chunk_texts:
                        continue
                        
                    all_transcript_segments.append(t_seg)
                    current_chunk_texts.add(txt)
                
                # Update history for next iteration
                last_chunk_texts = current_chunk_texts

        return {"segments_to_remove": all_remove_segments, "transcript_segments": all_transcript_segments}

    def _call_gemini_chunk(self, prompt, chunk_data, chunk_num, total_chunks) -> Dict:
        # Use dictionary mapping for robust configuration
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        for attempt in range(3):
            response = None
            try:
                with console.status(f"[bold cyan]Analyzing Text Chunk {chunk_num}/{total_chunks} (Attempt {attempt+1})...[/bold cyan]"):
                    response = self.model.generate_content(
                        [prompt, chunk_data],
                        generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192},
                        safety_settings=safety_settings,
                        stream=False
                    )
                    text = response.text.strip()
                    
                # Clean markdown
                if text.startswith("```json"): text = text[7:]
                if text.endswith("```"): text = text[:-3]
                
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        # Handle case where AI returned just the list of segments (common deviation)
                        return {"segments_to_remove": data, "transcript_segments": []}
                    return data
                except json.JSONDecodeError:
                    console.print(f"[yellow]Warning: Chunk {chunk_num} JSON malformed. Recovering...[/yellow]")
                    # Same recovery logic as before
                    segments = []
                    transcript_segs = []
                    seg_match = re.search(r'"segments_to_remove"\s*:\s*(\[.*?\])', text, re.DOTALL)
                    if seg_match:
                        try: segments = json.loads(seg_match.group(1))
                        except: pass
                    trans_match = re.search(r'"transcript_segments"\s*:\s*(\[.*?\])', text, re.DOTALL)
                    if trans_match: 
                        # Simple robust extraction for list
                        # Find the start [
                        s_idx = text.find('[', trans_match.start())
                        s = text[s_idx:]
                        # Try to fix truncated list
                        # Remove trailing "}" if present
                        if s.strip().endswith("}"): s = s.strip()[:-1]
                        
                        if not s.endswith("]"): 
                            last_brace = s.rfind("}")
                            if last_brace != -1: s = s[:last_brace+1] + "]"
                            else: s = "[]"

                        try: transcript_segs = json.loads(s)
                        except: pass
                    return {"segments_to_remove": segments, "transcript_segments": transcript_segs}
                    
            except Exception as e:
                console.print(f"[yellow]Error Chunk {chunk_num}: {repr(e)}[/yellow]")
                if response and hasattr(response, 'prompt_feedback'):
                    console.print(f"[red]Prompt Feedback: {response.prompt_feedback}[/red]")
                time.sleep(2)
                
        return {}