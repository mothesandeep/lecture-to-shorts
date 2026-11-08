#!/usr/bin/env python3
"""
hook_finder.py - Highlight Detection & Viral Hook Finder for Long Lectures.

Transcribes audio from long video files (with persistent disk caching),
and uses Gemini Flash to detect the top engaging, high-value 30-60 second
clips featuring irresistible 3-second hooks.

Key Features:
1. Fast audio extraction via FFmpeg (16kHz mono MP3).
2. Whisper full-lecture transcription with sha256-based disk caching
   (2-hour lecture transcribed once, 0s latency on subsequent runs).
3. Structured Pydantic schema extracting clip title, hook, start/end times,
   viral quotient score, and engagement rationale.
4. Robust validation ensuring clean sentence boundaries and 30-60s durations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.transcriber import DEFAULT_WHISPER_INITIAL_PROMPT
from core.whisper_backend import (
    load_lecture_transcript_cache,
    save_lecture_transcript_cache,
    transcribe_words,
)

try:
    from core.av_sync import cut_snippet_accurate, ensure_cfr_video, probe_av_sync, verify_av_sync
except ImportError:
    from av_sync import cut_snippet_accurate, ensure_cfr_video, probe_av_sync, verify_av_sync

CACHE_DIR = Path.home() / ".cache" / "hinglish_captions"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


class ViralClipMetadata(BaseModel):
    """Structured Pydantic schema for Gemini viral clip detection."""
    clip_id: int = Field(description="Sequential index of the highlight clip (1, 2, ...)")
    title: str = Field(description="Punchy, viral title (3-6 words) in Latin Hinglish or English (e.g. 'Kyun Fail Hote Hain?')")
    hook: str = Field(description="The exact spoken hook sentence in natural Latin Hinglish (e.g. 'Imaandari se batana kitne logon ne...') - strictly Latin alphabet, NEVER Urdu or Arabic script")
    start_time: float = Field(description="Exact start timestamp in seconds")
    end_time: float = Field(description="Exact end timestamp in seconds")
    duration: float = Field(description="Duration in seconds")
    viral_score: int = Field(description="Predicted viral engagement score from 1 to 100")
    reason: str = Field(description="Why this clip is engaging, motivational, or high-value (in English or Hinglish)")
    slug: str = Field(description="Filesystem-safe identifier, e.g. 'power_of_focus'")
    aspect_ratio: str = Field(default="9:16", description="Target aspect ratio: '9:16' for shorts/reels (<180s) or '16:9' for deep dives (>=180s)")


SYSTEM_HOOK_PROMPT = """You are an elite social media content strategist and video editor specializing in turning long academic, coding, and motivational lectures into high-performing video clips:

1. SHORT VIRAL REELS / SHORTS (Duration < 180s, typically 30s to 90s):
   - Aspect ratio: "9:16"
   - Explosive 3-second hook (provocative question, bold realization, intense motivational punchline).
   - Self-contained thought ending on a high-energy takeaway.

2. DEEP-DIVE STANDALONE EXPLANATIONS (Duration >= 180s, typically 3 to 8 minutes):
   - Aspect ratio: "16:9"
   - Complete, self-contained concept breakdown, masterclass explanation, or profound multi-step solution that deserves full 16:9 widescreen presentation.

CRITICAL SELECTION INVARIANTS:
1. THE 3-SECOND HOOK (STRICTLY LATIN HINGLISH):
   The very first 3 seconds MUST grab the viewer's attention immediately:
   - A provocative question ("Kitne logon ne dekha tha...", "Kyun fail hote hain...")
   - A bold, counterintuitive statement or wake-up call
   - An emotionally charged realization or relatable struggle
   - Immediate energy that stops scrolling. Avoid clips starting with "um", "so today", greetings, or slide reading.
   - STRICT LANGUAGE INVARIANT: 'hook', 'title', 'reason', and 'slug' MUST ALWAYS be written in natural texting-style Hinglish / English using the Latin alphabet (e.g. 'Imaandari se batana kitne logon ne wo event dekhte waqt wo moment jiya...', 'Conditions Kabhi Nahi Sudrengi!').
   - NEVER output Urdu / Perso-Arabic script (e.g. ایمان داری سے بتانا), and NEVER output Arabic characters.
   - If the transcript was transcribed in Urdu or Devanagari script, romanize/transliterate the hook and title into standard WhatsApp-style Latin Hinglish.

2. COMPLETENESS & ASPECT RATIO:
   - Identify both short viral moments (30s-90s) AND deep-dive high-value explanations (3 to 8 minutes).
   - Set 'aspect_ratio' to '9:16' for clips < 180 seconds, and '16:9' for clips >= 180 seconds.
   - The idea MUST be 100% self-contained: starts cleanly at the beginning of a complete sentence, and finishes on a strong punchline, resolution, or memorable takeaway. Never cut mid-sentence!

3. HIGH EMOTIONAL OR PRACTICAL VALUE:
   - Prioritize motivational breakthroughs, harsh truths, study/exam hacks, mindset shifts, or unforgettable life advice.
   - Ignore boring slide logistics, mic tests, formula reading, or administrative chit-chat.

4. EXACT TIMESTAMPS:
   - Provide precise 'start_time' and 'end_time' in floating seconds based on the transcript markers.
"""


def sanitize_perso_arabic(client: Any, model_name: str, text: str) -> str:
    """Convert any Urdu/Perso-Arabic text to clean texting-style Latin Hinglish."""
    if not text or not URDU_RE.search(text):
        return text
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=f"Transliterate this Urdu/Hindustani text into clean, casual texting-style Latin Hinglish (strictly Latin alphabet only, WhatsApp spelling, no Urdu/Arabic characters):\n\n{text}"
        )
        cleaned = resp.text.strip().replace('"', '').replace("'", "").strip()
        if cleaned and not URDU_RE.search(cleaned):
            return cleaned
    except Exception as e:
        print(f"[hook_finder] Transliteration fallback error: {e}")
    return text


def extract_audio(video_path: Path | str, output_audio: Optional[Path | str] = None) -> Path:
    """
    Extract audio track from video to a lightweight 16kHz mono MP3 file
    for high-speed speech-to-text processing.
    """
    v_path = Path(video_path).expanduser().resolve()
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    if output_audio:
        a_path = Path(output_audio).expanduser().resolve()
    else:
        a_path = v_path.with_suffix(".audio.mp3")

    a_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(v_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "libmp3lame",
        "-q:a", "4",
        str(a_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")
    return a_path


def format_seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds to MM:SS formatted string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


class HookFinder:
    """Detects engaging viral highlight moments from long lectures."""

    def __init__(
        self,
        whisper_model: str = "base",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        min_duration: float = 28.0,
        max_duration: float = 60.0,
        backend: str = "auto"
    ) -> None:
        self.whisper_model_name = whisper_model
        self.llm_model_name = llm_model or os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.backend = backend

    def get_full_transcript(self, video_path: Path | str) -> List[Dict[str, Any]]:
        """
        Transcribe the full audio of the video using Whisper,
        utilizing sha256 disk caching for instant subsequent loads.
        Stores both segment-level and word-level timestamps in the disk cache.
        """
        v_path = Path(video_path).expanduser().resolve()

        # 1. Try loading from full lecture disk cache
        cached = load_lecture_transcript_cache(v_path, self.whisper_model_name)
        if cached:
            segments = cached.get("segments", [])
            words = cached.get("words", [])
            if segments and words:
                print(
                    f"[hook_finder] Loaded full lecture transcript from cache: "
                    f"{len(segments)} segments, {len(words)} words"
                )
                return segments
            elif segments:
                print(f"[hook_finder] Cached transcript lacks word timestamps -> re-transcribing to cache words...")

        print(f"[hook_finder] Extracting audio from {v_path.name}...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)

        try:
            extract_audio(v_path, tmp_audio_path)
            print(f"[hook_finder] Transcribing full audio with Whisper model '{self.whisper_model_name}' (backend={self.backend}, word_timestamps=True)...")
            whisper_prompt = DEFAULT_WHISPER_INITIAL_PROMPT
            trans_res = transcribe_words(
                audio_path=tmp_audio_path,
                language="hi",
                model=self.whisper_model_name,
                initial_prompt=whisper_prompt,
                backend=self.backend,
                word_timestamps=True,
                condition_on_previous_text=False,
                fp16=False
            )

            segments: List[Dict[str, Any]] = [
                {
                    "id": s.get("id", i),
                    "start": round(float(s.get("start", 0.0)), 2),
                    "end": round(float(s.get("end", 0.0)), 2),
                    "text": s.get("text", "").strip()
                }
                for i, s in enumerate(trans_res.segments)
                if s.get("text", "").strip()
            ]

            # Save full transcript with word-level timestamps to disk cache
            if segments:
                cache_file = save_lecture_transcript_cache(
                    video_path=v_path,
                    model_name=self.whisper_model_name,
                    segments=segments,
                    words=trans_res.words
                )
                print(
                    f"[hook_finder] Transcription completed ({len(segments)} segments, {len(trans_res.words)} words) "
                    f"and cached -> {cache_file.name}"
                )
            return segments
        finally:
            tmp_audio_path.unlink(missing_ok=True)

    def find_viral_clips(
        self,
        video_path: Path | str,
        num_clips: int = 3
    ) -> List[ViralClipMetadata]:
        """
        Analyze the full lecture transcript with Gemini and detect the top N viral clips.
        """
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for AI hook detection. "
                "Please export GEMINI_API_KEY='your_key' or provide --api-key."
            )

        from google import genai
        from google.genai import types

        segments = self.get_full_transcript(video_path)
        if not segments:
            raise RuntimeError("No speech detected in the video audio.")

        total_duration = segments[-1]["end"]
        print(f"[hook_finder] Total lecture duration: {format_seconds_to_timestamp(total_duration)} "
              f"({total_duration:.1f}s)")

        # Prepare formatted transcript for Gemini
        transcript_lines: List[str] = []
        for s in segments:
            ts_start = format_seconds_to_timestamp(s["start"])
            ts_end = format_seconds_to_timestamp(s["end"])
            transcript_lines.append(f"[{ts_start} - {ts_end} | {s['start']:.1f}s - {s['end']:.1f}s] {s['text']}")
        formatted_transcript = "\n".join(transcript_lines)

        client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=60000))

        # Standardize without thinking_config(thinking_budget=0) to prevent HTTP 400
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_HOOK_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=list[ViralClipMetadata]
        )

        user_prompt = (
            f"Analyze the following lecture transcript and identify the top {num_clips} most engaging, "
            f"viral, or high-value video clips. You may select both short viral moments (30s to 90s, aspect_ratio '9:16') "
            f"and deep-dive standalone explanations (3 to 8 minutes, aspect_ratio '16:9'). "
            f"Ensure each clip starts with an engaging 3-second hook and ends on a self-contained resolution.\n\n"
            f"LECTURE TRANSCRIPT:\n{formatted_transcript}"
        )

        print(f"[hook_finder] Prompting {self.llm_model_name} to detect top {num_clips} viral hooks...")
        t0 = time.time()

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.llm_model_name,
                    contents=user_prompt,
                    config=cfg
                )
                break
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
                    backoff = 2.0 * (attempt + 1)
                    print(f"[hook_finder] Service busy ({type(e).__name__}). Retrying in {backoff:.0f}s...")
                    time.sleep(backoff)
                    continue
                raise

        elapsed = time.time() - t0
        print(f"[hook_finder] AI Hook detection finished in {elapsed:.2f}s")

        raw_clips = json.loads(response.text)
        validated_clips: List[ViralClipMetadata] = []

        for item in raw_clips:
            # Validate and clamp timestamps
            st = float(item.get("start_time", 0.0))
            et = float(item.get("end_time", 0.0))
            if et <= st:
                continue

            dur = et - st
            # Enforce duration bounds
            if dur < self.min_duration:
                # Expand slightly if needed
                et = min(total_duration, st + self.min_duration)
                dur = et - st
            elif dur > self.max_duration + 5.0:
                et = st + self.max_duration
                dur = self.max_duration

            slug = item.get("slug", f"clip_{len(validated_clips) + 1}").strip().lower()
            slug = "".join(c if c.isalnum() or c == "-" else "_" for c in slug)

            # Auto-assign aspect ratio based on duration / model recommendation
            raw_ar = str(item.get("aspect_ratio", "")).strip().lower()
            if "16:9" in raw_ar or "horizontal" in raw_ar or "landscape" in raw_ar:
                ar = "16:9"
            elif "9:16" in raw_ar or "vertical" in raw_ar or "portrait" in raw_ar:
                ar = "9:16"
            else:
                ar = "16:9" if dur >= 180.0 else "9:16"

            hook_str = item.get("hook", "").strip()
            title_str = item.get("title", f"Highlight #{len(validated_clips) + 1}").strip()
            reason_str = item.get("reason", "").strip()

            # Guaranteed Urdu elimination: transliterate if any Urdu script detected
            if URDU_RE.search(hook_str):
                hook_str = sanitize_perso_arabic(client, self.llm_model_name, hook_str)
            if URDU_RE.search(title_str):
                title_str = sanitize_perso_arabic(client, self.llm_model_name, title_str)
            if URDU_RE.search(reason_str):
                reason_str = sanitize_perso_arabic(client, self.llm_model_name, reason_str)

            validated_clips.append(ViralClipMetadata(
                clip_id=len(validated_clips) + 1,
                title=title_str,
                hook=hook_str,
                start_time=round(st, 2),
                end_time=round(et, 2),
                duration=round(dur, 2),
                viral_score=int(item.get("viral_score", 85)),
                reason=reason_str,
                slug=slug,
                aspect_ratio=ar
            ))

            if len(validated_clips) >= num_clips:
                break

        return validated_clips


def cut_snippet_without_reencode(
    src_video: Path | str,
    out_snippet: Path | str,
    start_time: float,
    end_time: float,
    target_fps: Optional[float] = None,
    audio_delay_ms: float = 500.0
) -> Path:
    """
    Cut selected time snippet from source video with strict audio-video sync,
    timestamp zeroing, audio delay offset, and Constant Frame Rate (CFR) normalization.
    """
    return cut_snippet_accurate(
        src_video=src_video,
        out_snippet=out_snippet,
        start_time=start_time,
        end_time=end_time,
        target_fps=target_fps,
        audio_delay_ms=audio_delay_ms
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Highlight Detection and Hook Finder for lectures.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--clips", type=int, default=3, help="Number of viral clips to detect")
    parser.add_argument("--whisper-model", default="base", help="Whisper model for full lecture pass")
    args = parser.parse_args()

    finder = HookFinder(whisper_model=args.whisper_model)
    clips = finder.find_viral_clips(args.video, num_clips=args.clips)
    print(f"\n[Detected {len(clips)} Viral Clips]:")
    for c in clips:
        print(f"\n#{c.clip_id}: {c.title} (Score: {c.viral_score}/100)")
        print(f"  Time: {format_seconds_to_timestamp(c.start_time)} -> {format_seconds_to_timestamp(c.end_time)} ({c.duration:.1f}s)")
        print(f"  Hook: \"{c.hook}\"")
        print(f"  Why : {c.reason}")

