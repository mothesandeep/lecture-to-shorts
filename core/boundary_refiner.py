#!/usr/bin/env python3
"""
boundary_refiner.py - High-Precision Sentence Boundary Re-verification (+/- 60s Window).

Inspects audio within a +/- 60s context window around rough timestamps,
runs a fast Whisper word-level pass, and snaps start and end boundaries to natural
sentence silences and punchlines so speech is never clipped mid-word or mid-thought.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_video_duration(video_path: Path | str) -> float:
    """Probe total duration of video in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 999999.0


def format_timestamp_ms(seconds: float) -> str:
    """Convert float seconds to MM:SS.S format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:04.1f}"


def refine_clip_boundaries(
    video_path: Path | str,
    start_t: float,
    end_t: float,
    margin_s: float = 60.0,
    whisper_model: str = "base"
) -> Tuple[float, float]:
    """
    Refine rough clip start and end timestamps using a +/- margin_s audio window:
    - Finds the natural starting pause/silence closest to start_t so the speaker's
      first word isn't clipped mid-sentence.
    - Finds the natural ending pause right after the punchline closest to end_t so
      sentences don't cut off abruptly.

    Returns:
        Tuple of (refined_start, refined_end) in float seconds.
    """
    v_path = Path(video_path).expanduser().resolve()
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found: {v_path}")

    total_dur = get_video_duration(v_path)
    window_start = max(0.0, start_t - margin_s)
    window_end = min(total_dur, end_t + margin_s)
    window_dur = window_end - window_start

    if window_dur <= 5.0 or start_t >= end_t:
        return start_t, end_t

    print(f"\n[boundary] High-Precision Boundary Re-verification (+/- {margin_s:.0f}s window)...")
    print(f"[boundary] Probing audio context [{format_timestamp_ms(window_start)} -> {format_timestamp_ms(window_end)}] ({window_dur:.1f}s)...")

    # 1. Extract lightweight mono audio snippet for the window
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_audio = Path(tmp.name)

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{window_start:.3f}",
            "-t", f"{window_dur:.3f}",
            "-i", str(v_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "libmp3lame",
            "-q:a", "4",
            str(tmp_audio)
        ]
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[boundary] Audio extraction failed: {res.stderr} -> using original timestamps.")
            return start_t, end_t

        # 2. Fast Whisper word-level pass on the 2-3 minute audio chunk
        import whisper
        wm = whisper.load_model(whisper_model)
        transcribe_res = wm.transcribe(
            str(tmp_audio),
            language="hi",
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.3,
            fp16=False
        )

        # 3. Collect word timestamps shifted to global video timeline
        words: List[Dict[str, Any]] = []
        terminal_puncts = {".", "!", "?", "।", "॥"}

        for segment in transcribe_res.get("segments", []):
            for w in segment.get("words", []):
                cleaned = w.get("word", "").strip()
                if cleaned:
                    w_start = window_start + float(w.get("start", 0.0))
                    w_end = window_start + float(w.get("end", 0.0))
                    words.append({
                        "word": cleaned,
                        "start": w_start,
                        "end": w_end
                    })

        if not words:
            print("[boundary] No speech detected in boundary window -> retaining original timestamps.")
            return start_t, end_t

        # 4. Snap Starting Boundary
        # Look for natural sentence starts within [start_t - 15.0, start_t + 8.0]
        start_candidates: List[Dict[str, Any]] = []

        for i, w in enumerate(words):
            # Check if this word could be a sentence start
            w_start = w["start"]
            if w_start < start_t - 18.0 or w_start > start_t + 10.0:
                continue

            pause_before = 999.0 if i == 0 else (w_start - words[i - 1]["end"])
            prev_ended_sentence = (i > 0) and any(words[i - 1]["word"].endswith(p) for p in terminal_puncts)

            is_sentence_start = (pause_before >= 0.30) or prev_ended_sentence or (i == 0)

            # Score candidate: prefer natural pause closest to start_t without cutting speech
            if is_sentence_start:
                if w_start <= start_t:
                    dist_penalty = (start_t - w_start) * 0.7  # Small penalty for slight lead-in
                else:
                    dist_penalty = (w_start - start_t) * 1.6  # Heavier penalty for clipping hook

                pause_bonus = -min(2.0, pause_before * 1.5) if pause_before < 10.0 else -1.0
                score = dist_penalty + pause_bonus
                start_candidates.append({
                    "start": max(0.0, w_start - 0.12),  # 120ms buffer for plosive consonants
                    "score": score,
                    "word": w["word"]
                })

        if start_candidates:
            start_candidates.sort(key=lambda c: c["score"])
            refined_start = round(start_candidates[0]["start"], 2)
        else:
            # If no clear sentence break, snap to closest word start
            closest_word = min(words, key=lambda w: abs(w["start"] - start_t))
            refined_start = round(max(0.0, closest_word["start"] - 0.10), 2)

        # 5. Snap Ending Boundary
        # Look for natural punchline / sentence completion in [end_t - 8.0, end_t + 15.0]
        end_candidates: List[Dict[str, Any]] = []

        for i, w in enumerate(words):
            w_end = w["end"]
            if w_end < end_t - 10.0 or w_end > end_t + 18.0:
                continue

            pause_after = 999.0 if i == len(words) - 1 else (words[i + 1]["start"] - w_end)
            ends_sentence = any(w["word"].endswith(p) for p in terminal_puncts)

            is_sentence_end = (pause_after >= 0.35) or ends_sentence or (i == len(words) - 1)

            if is_sentence_end:
                if w_end >= end_t:
                    dist_penalty = (w_end - end_t) * 0.7  # Small penalty for allowing punchline resolution
                else:
                    dist_penalty = (end_t - w_end) * 1.5  # Heavier penalty for premature cut

                pause_bonus = -min(2.0, pause_after * 1.5) if pause_after < 10.0 else -1.0
                score = dist_penalty + pause_bonus
                end_candidates.append({
                    "end": min(total_dur, w_end + 0.18),  # 180ms buffer for natural breath decay
                    "score": score,
                    "word": w["word"]
                })

        if end_candidates:
            end_candidates.sort(key=lambda c: c["score"])
            refined_end = round(end_candidates[0]["end"], 2)
        else:
            # If no clear break, snap to closest word end
            closest_word = min(words, key=lambda w: abs(w["end"] - end_t))
            refined_end = round(min(total_dur, closest_word["end"] + 0.15), 2)

        # Enforce sanity minimum duration
        if refined_end <= refined_start + 5.0:
            refined_start = start_t
            refined_end = end_t

        diff_s = (refined_end - refined_start) - (end_t - start_t)
        sign = "+" if diff_s >= 0 else ""
        print(f"[boundary] Refined cut from [{format_timestamp_ms(start_t)} -> {format_timestamp_ms(end_t)}] to "
              f"[{format_timestamp_ms(refined_start)} -> {format_timestamp_ms(refined_end)}] "
              f"({sign}{diff_s:.1f}s boundary adjusted for natural sentence completion)")

        return refined_start, refined_end

    finally:
        if tmp_audio.exists():
            tmp_audio.unlink(missing_ok=True)

