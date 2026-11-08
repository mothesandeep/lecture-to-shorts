#!/usr/bin/env python3
"""
hook_finder.py - Highlight Detection & Viral Hook Finder for Long Lectures.

Transcribes audio from long video files (with persistent disk caching),
extracts acoustic prosody & pacing tags, and uses Gemini Flash to detect
the top engaging, high-value clips featuring irresistible 3-second hooks.

Key Enhancements:
1. Segment-based prompt numbering [S0001]... with code-level timestamp conversion.
2. Acoustic prosody & speech pacing integration (RMS energy z-score, words/sec, pause density).
3. Weighted sub-score evaluation: hook_strength, standalone, emotion, payoff, shareability.
4. Map-reduce pipeline for long multi-hour lectures (10-min sliding windows with overlap & global rerank).
5. Strict anti-pattern filtering (slow intros, "as I said before", "look at the board").
6. Optional face-presence filtering for 9:16 vertical reels (>=70% teacher visibility).
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from core.audio_features import AudioFeatureExtractor
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

DEFAULT_VIRAL_SCORE_WEIGHTS: Dict[str, float] = {
    "hook_strength": 0.30,   # Scroll-stopping power of first 3s
    "standalone": 0.20,      # Self-contained understanding without context
    "emotion": 0.20,         # Emotional resonance & vulnerability
    "payoff": 0.15,          # Quality and insight of the punchline
    "shareability": 0.15,    # Willingness of viewer to share/save
}


class ViralClipCandidate(BaseModel):
    """Structured Pydantic schema for Gemini segment-based viral clip proposal."""
    start_segment_id: str = Field(description="Starting segment ID from transcript, e.g. 'S0001'")
    end_segment_id: str = Field(description="Ending segment ID from transcript, e.g. 'S0014'")
    title: str = Field(description="Punchy, viral title (3-6 words) in Latin Hinglish or English (e.g. 'Kyun Fail Hote Hain?')")
    hook: str = Field(description="The exact spoken hook sentence in natural Latin Hinglish (strictly Latin alphabet, NEVER Urdu or Arabic script)")
    aspect_ratio: str = Field(default="9:16", description="Target aspect ratio: '9:16' for shorts/reels (<180s) or '16:9' for deep dives (>=180s)")
    hook_strength: int = Field(description="Score 1-10: How irresistibly scroll-stopping the first 3 seconds are")
    standalone: int = Field(description="Score 1-10: How self-contained and complete the idea is without prior context")
    emotion: int = Field(description="Score 1-10: Emotional resonance, passion, vulnerability, or motivational intensity")
    payoff: int = Field(description="Score 1-10: Satisfaction, wisdom, and value delivered by the resolution")
    shareability: int = Field(description="Score 1-10: Likelihood of viewers sending this clip to friends or saving it")
    reason: str = Field(description="Why this clip is engaging, motivational, or high-value (in English or Latin Hinglish)")
    slug: str = Field(description="Filesystem-safe identifier, e.g. 'power_of_focus'")


class ViralClipMetadata(BaseModel):
    """Structured clip metadata representing a ready-to-cut highlight clip."""
    clip_id: int = Field(description="Sequential index of the highlight clip (1, 2, ...)")
    title: str = Field(description="Punchy, viral title (3-6 words) in Latin Hinglish or English")
    hook: str = Field(description="The exact spoken hook sentence in natural Latin Hinglish")
    start_time: float = Field(description="Exact start timestamp in seconds")
    end_time: float = Field(description="Exact end timestamp in seconds")
    duration: float = Field(description="Duration in seconds")
    viral_score: int = Field(description="Composite viral engagement score from 1 to 100")
    reason: str = Field(description="Why this clip is engaging, motivational, or high-value")
    slug: str = Field(description="Filesystem-safe identifier, e.g. 'power_of_focus'")
    aspect_ratio: str = Field(default="9:16", description="Target aspect ratio: '9:16' or '16:9'")
    start_segment_id: Optional[str] = Field(default=None, description="Starting segment identifier (e.g. 'S0001')")
    end_segment_id: Optional[str] = Field(default=None, description="Ending segment identifier (e.g. 'S0014')")
    sub_scores: Optional[Dict[str, float]] = Field(default=None, description="Individual viral sub-scores (1-10)")


class RerankSelection(BaseModel):
    """Structured Pydantic schema for global candidate reranking."""
    selected_slugs: List[str] = Field(description="Ordered list of chosen candidate clip slugs by viral quality and topic diversity")


SYSTEM_HOOK_PROMPT = """You are an elite social media content strategist and video editor specializing in turning long academic, coding, and motivational lectures into high-performing viral video clips:

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
   - Immediate energy that stops scrolling.
   - STRICT LANGUAGE INVARIANT: 'hook', 'title', 'reason', and 'slug' MUST ALWAYS be written in natural texting-style Hinglish / English using the Latin alphabet (e.g. 'Imaandari se batana kitne logon ne wo event dekhte waqt wo moment jiya...', 'Conditions Kabhi Nahi Sudrengi!').
   - NEVER output Urdu / Perso-Arabic script (e.g. ایمان داری سے بتانا), and NEVER output Arabic characters.
   - If the transcript was transcribed in Urdu or Devanagari script, romanize/transliterate the hook and title into standard WhatsApp-style Latin Hinglish.

2. ACOUSTIC & PACING TAGS:
   - Each transcript line is tagged with [energy +X.Xσ] and pacing tags like [fast], [slow], [punchy], [pauses].
   - Give strong preference to moments tagged with positive energy (+1.0σ to +3.0σ) and [fast] or [punchy] pacing.
   - Avoid monotonous sections with negative energy or heavy pauses.

3. STRICT ANTI-PATTERNS (DO NOT SELECT CLIPS CONTAINING THESE):
   - SLOW INTROS & CHIT-CHAT: Greetings, mic tests, "good morning", "haan toh bacho", "welcome back", "so today we will discuss".
   - BACKWARD REFERENCES: "As I said before", "jaise maine pehle bataya tha", "in the previous lecture", "last class mein humne dekha". The clip must NOT feel like the continuation of a previous topic.
   - VISUAL-DEPENDENT REFERENCES: "Look at the board", "board pe dekho", "look at page 45", "in this diagram", "formula number 3". If a viewer cannot understand the clip just by listening and looking at the speaker's face, DO NOT SELECT IT.
   - INCOMPLETE THOUGHTS: Cutting off mid-sentence, rhetorical pauses that resolve later, or concepts left unanswered.

4. SEGMENT-BASED IDENTIFIERS:
   - Each transcript line begins with a segment ID like [S0001], [S0002], etc.
   - You MUST output 'start_segment_id' and 'end_segment_id' (e.g. start_segment_id="S0001", end_segment_id="S0014").
   - Rate the clip across all 5 sub-scores from 1 to 10 (hook_strength, standalone, emotion, payoff, shareability).
"""


def normalize_segment_id(raw_id: Any) -> Optional[str]:
    """Normalize raw segment id (e.g. 'S0001', '[S0001]', 'S1', 1) into standard 'S0001' format."""
    if raw_id is None:
        return None
    if isinstance(raw_id, int):
        return f"S{raw_id:04d}"
    clean = str(raw_id).strip().strip("[]").strip()
    m = re.search(r"(\d+)", clean)
    if m:
        return f"S{int(m.group(1)):04d}"
    return None


def compute_weighted_viral_score(
    sub_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> int:
    """
    Compute composite viral engagement score (1 to 100) from sub-scores (1 to 10).
    """
    eff_weights = weights or DEFAULT_VIRAL_SCORE_WEIGHTS
    w_sum = sum(eff_weights.values()) or 1.0
    total = 0.0
    for k, w in eff_weights.items():
        val = float(sub_scores.get(k, 7.0))
        if val > 10.0:  # normalize 1-100 to 1-10 if needed
            val = val / 10.0
        total += (w / w_sum) * val
    return max(1, min(100, int(round(total * 10.0))))


def deduplicate_candidate_clips(
    clips: List[ViralClipMetadata],
    overlap_ratio_thr: float = 0.40,
    start_diff_thr: float = 10.0
) -> List[ViralClipMetadata]:
    """
    Deduplicate overlapping candidate clips from adjacent map windows.
    Keeps the candidate with higher viral_score when two clips overlap significantly.
    """
    if not clips:
        return []

    sorted_clips = sorted(
        clips,
        key=lambda c: (c.viral_score, getattr(c, "sub_scores", {}).get("hook_strength", 0) if c.sub_scores else 0),
        reverse=True
    )
    kept: List[ViralClipMetadata] = []

    for cand in sorted_clips:
        is_duplicate = False
        for existing in kept:
            overlap = max(0.0, min(cand.end_time, existing.end_time) - max(cand.start_time, existing.start_time))
            min_dur = min(cand.duration, existing.duration)
            start_diff = abs(cand.start_time - existing.start_time)

            if min_dur > 0 and (overlap / min_dur > overlap_ratio_thr or start_diff < start_diff_thr):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(cand)

    return kept


def check_face_presence(
    video_path: Path | str,
    start_t: float,
    end_t: float,
    min_face_ratio: float = 0.70
) -> Tuple[bool, float]:
    """
    Verify presenter face visibility on a candidate vertical snippet.
    Reuses OpenCV Haar cascade face detector on sampled downscaled frames.
    Returns (passes_filter, face_presence_ratio).
    """
    v_p = Path(video_path).expanduser().resolve()
    if not v_p.is_file():
        return True, 1.0

    import cv2
    from core.reframer import find_cascade

    try:
        cascade_file = find_cascade()
        face_cascade = cv2.CascadeClassifier(str(cascade_file))
        if face_cascade.empty():
            return True, 1.0

        cap = cv2.VideoCapture(str(v_p))
        if not cap.isOpened():
            return True, 1.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_frame = int(round(start_t * fps))
        end_frame = int(round(end_t * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        checked = 0
        found = 0
        step = max(1, int(round(fps * 0.16)))  # Sample every ~6 frames (~4 checks per second)

        current_frame = start_frame
        while current_frame <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (current_frame - start_frame) % step == 0:
                h, w = frame.shape[:2]
                small_w = 320
                small_h = max(1, int(round(h * small_w / w)))
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                faces = face_cascade.detectMultiScale(small, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
                checked += 1
                if len(faces) > 0:
                    found += 1
            current_frame += 1

        cap.release()

        ratio = (found / checked) if checked > 0 else 1.0
        return (ratio >= min_face_ratio), ratio
    except Exception as e:
        print(f"[hook_finder] Face presence check warning: {e}")
        return True, 1.0


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
        backend: str = "auto",
        weights: Optional[Dict[str, float]] = None,
        filter_face_presence: bool = False,
        map_reduce_threshold: float = 600.0,
    ) -> None:
        self.whisper_model_name = whisper_model
        self.llm_model_name = llm_model or os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.backend = backend
        self.weights = weights or DEFAULT_VIRAL_SCORE_WEIGHTS
        self.filter_face_presence = filter_face_presence
        self.map_reduce_threshold = map_reduce_threshold
        self.last_cached_words: List[Any] = []

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
                self.last_cached_words = words
                # Ensure each segment has standardized seg_id
                for i, s in enumerate(segments):
                    if "seg_id" not in s:
                        s["seg_id"] = f"S{i + 1:04d}"
                    if "id" not in s:
                        s["id"] = i + 1
                print(
                    f"[hook_finder] Loaded full lecture transcript from cache: "
                    f"{len(segments)} segments, {len(words)} words"
                )
                return segments
            elif segments:
                print("[hook_finder] Cached transcript lacks word timestamps -> re-transcribing to cache words...")

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
                    "id": i + 1,
                    "seg_id": f"S{i + 1:04d}",
                    "start": round(float(s.get("start", 0.0)), 2),
                    "end": round(float(s.get("end", 0.0)), 2),
                    "text": s.get("text", "").strip()
                }
                for i, s in enumerate(trans_res.segments)
                if s.get("text", "").strip()
            ]
            self.last_cached_words = trans_res.words

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

    def _call_gemini_candidates(
        self,
        client: Any,
        formatted_transcript: str,
        num_clips: int
    ) -> List[ViralClipCandidate]:
        """Send formatted transcript chunk to Gemini and receive ViralClipCandidate objects."""
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_HOOK_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=list[ViralClipCandidate]
        )

        user_prompt = (
            f"Analyze the following lecture transcript lines and identify up to {num_clips} of the top most engaging, "
            f"viral, or high-value video clips. Each clip must have a duration between {self.min_duration:.0f}s and {self.max_duration:.0f}s. "
            f"Select exact start_segment_id and end_segment_id for each clip, and score the 5 sub-metrics (1-10).\n\n"
            f"LECTURE TRANSCRIPT:\n{formatted_transcript}"
        )

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

        raw_items = json.loads(response.text)
        candidates: List[ViralClipCandidate] = []
        for item in raw_items:
            try:
                candidates.append(ViralClipCandidate(**item))
            except Exception as e:
                print(f"[hook_finder] Warning: Skipping malformed candidate: {e}")
        return candidates

    def _rerank_candidates_gemini(
        self,
        client: Any,
        candidates: List[ViralClipMetadata],
        num_clips: int
    ) -> List[ViralClipMetadata]:
        """Global rerank call asking Gemini to select the top num_clips from candidate pool."""
        from google.genai import types

        cand_summaries = []
        slug_to_cand = {c.slug: c for c in candidates}

        for i, c in enumerate(candidates, 1):
            cand_summaries.append(
                f"[{i}] Slug: '{c.slug}' | Duration: {c.duration:.1f}s | Aspect: {c.aspect_ratio} | "
                f"Score: {c.viral_score}/100\n"
                f"    Title: {c.title}\n"
                f"    Hook: \"{c.hook}\"\n"
                f"    Why: {c.reason}\n"
            )
        pool_text = "\n".join(cand_summaries)

        rerank_prompt = (
            f"You are the chief content editor. Below are {len(candidates)} candidate clips extracted "
            f"from across an entire lecture. Rank and select the top {num_clips} absolute best clips "
            f"based on maximum viral potential, hook punchiness, emotional value, and topic diversity.\n\n"
            f"CANDIDATE POOL:\n{pool_text}"
        )

        cfg = types.GenerateContentConfig(
            system_instruction="Select the most viral, diverse clips and output their slugs in ranked order.",
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=RerankSelection
        )

        try:
            print(f"[hook_finder] Global rerank: Selecting top {num_clips} clips from {len(candidates)} candidates...")
            resp = client.models.generate_content(
                model=self.llm_model_name,
                contents=rerank_prompt,
                config=cfg
            )
            selection = RerankSelection(**json.loads(resp.text))
            reranked: List[ViralClipMetadata] = []
            for slug in selection.selected_slugs:
                if slug in slug_to_cand and slug_to_cand[slug] not in reranked:
                    reranked.append(slug_to_cand[slug])
                if len(reranked) >= num_clips:
                    break

            # If rerank selected fewer than num_clips, fill from remaining sorted by score
            if len(reranked) < num_clips:
                for c in candidates:
                    if c not in reranked:
                        reranked.append(c)
                    if len(reranked) >= num_clips:
                        break
            return reranked
        except Exception as e:
            print(f"[hook_finder] Rerank fallback to python score sorting: {e}")
            return sorted(candidates, key=lambda c: c.viral_score, reverse=True)[:num_clips]

    def _convert_candidate_to_metadata(
        self,
        item: ViralClipCandidate,
        segment_map: Dict[str, Dict[str, Any]],
        total_duration: float,
        client: Optional[Any] = None
    ) -> Optional[ViralClipMetadata]:
        """Convert a ViralClipCandidate (with segment IDs) into a concrete ViralClipMetadata (with seconds)."""
        start_id = normalize_segment_id(item.start_segment_id)
        end_id = normalize_segment_id(item.end_segment_id)

        if not start_id or not end_id:
            return None

        start_seg = segment_map.get(start_id)
        end_seg = segment_map.get(end_id)

        if not start_seg or not end_seg:
            # Fallback to numerical index lookup if string ID didn't match directly
            all_ids = sorted(segment_map.keys())
            if not all_ids:
                return None
            start_seg = segment_map.get(start_id, segment_map[all_ids[0]])
            end_seg = segment_map.get(end_id, segment_map[all_ids[-1]])

        st = float(start_seg["start"])
        et = float(end_seg["end"])

        if et <= st:
            et = min(total_duration, st + self.min_duration)

        dur = et - st
        if dur < self.min_duration:
            et = min(total_duration, st + self.min_duration)
            dur = et - st
        elif dur > self.max_duration + 5.0:
            et = st + self.max_duration
            dur = self.max_duration

        sub_scores = {
            "hook_strength": float(item.hook_strength),
            "standalone": float(item.standalone),
            "emotion": float(item.emotion),
            "payoff": float(item.payoff),
            "shareability": float(item.shareability),
        }
        viral_score = compute_weighted_viral_score(sub_scores, self.weights)

        raw_slug = item.slug.strip().lower()
        slug = "".join(c if c.isalnum() or c == "-" else "_" for c in raw_slug)

        raw_ar = str(item.aspect_ratio).strip().lower()
        if "16:9" in raw_ar or "horizontal" in raw_ar or "landscape" in raw_ar:
            ar = "16:9"
        elif "9:16" in raw_ar or "vertical" in raw_ar or "portrait" in raw_ar:
            ar = "9:16"
        else:
            ar = "16:9" if dur >= 180.0 else "9:16"

        hook_str = item.hook.strip()
        title_str = item.title.strip()
        reason_str = item.reason.strip()

        if client is not None:
            if URDU_RE.search(hook_str):
                hook_str = sanitize_perso_arabic(client, self.llm_model_name, hook_str)
            if URDU_RE.search(title_str):
                title_str = sanitize_perso_arabic(client, self.llm_model_name, title_str)
            if URDU_RE.search(reason_str):
                reason_str = sanitize_perso_arabic(client, self.llm_model_name, reason_str)

        return ViralClipMetadata(
            clip_id=1,
            title=title_str,
            hook=hook_str,
            start_time=round(st, 2),
            end_time=round(et, 2),
            duration=round(dur, 2),
            viral_score=viral_score,
            reason=reason_str,
            slug=slug,
            aspect_ratio=ar,
            start_segment_id=start_id,
            end_segment_id=end_id,
            sub_scores=sub_scores,
        )

    def find_viral_clips(
        self,
        video_path: Path | str,
        num_clips: int = 3
    ) -> List[ViralClipMetadata]:
        """
        Analyze lecture audio and transcript to detect top N viral clips:
        - Annotates lines with acoustic prosody & pacing tags.
        - Uses Map-Reduce windows for lectures longer than map_reduce_threshold (default 10 mins).
        - Computes composite viral score via Python weights.
        - Optionally filters vertical candidates with < 70% presenter face presence.
        """
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for AI hook detection. "
                "Please export GEMINI_API_KEY='your_key' or provide --api-key."
            )

        from google import genai
        from google.genai import types

        v_path = Path(video_path).expanduser().resolve()
        segments = self.get_full_transcript(v_path)
        if not segments:
            raise RuntimeError("No speech detected in the video audio.")

        total_duration = segments[-1]["end"]
        print(f"[hook_finder] Total lecture duration: {format_seconds_to_timestamp(total_duration)} "
              f"({total_duration:.1f}s)")

        # 1. Extract audio features & annotate transcript lines
        try:
            audio_extractor = AudioFeatureExtractor.from_file(v_path)
            print(f"[hook_finder] Extracted audio energy profile: {len(audio_extractor.window_centers)} windows "
                  f"(mean RMS: {audio_extractor.mean_rms:.3f}, std: {audio_extractor.std_rms:.3f})")
        except Exception as exc:
            print(f"[hook_finder] Note: Audio feature extraction fallback: {exc}")
            audio_extractor = AudioFeatureExtractor()

        segment_map: Dict[str, Dict[str, Any]] = {}
        annotated_lines: List[str] = []
        for s in segments:
            seg_id = s.get("seg_id") or f"S{s.get('id', 1):04d}"
            s["seg_id"] = seg_id
            segment_map[seg_id] = s

            line = audio_extractor.format_annotated_segment(
                seg_id_str=f"[{seg_id}]",
                start_t=s["start"],
                end_t=s["end"],
                text=s["text"],
                words=self.last_cached_words
            )
            annotated_lines.append(line)

        client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=60000))
        t0 = time.time()

        raw_candidates: List[ViralClipCandidate] = []

        # 2. Map-Reduce for long lectures vs Single Call for short lectures
        use_map_reduce = total_duration > self.map_reduce_threshold

        if use_map_reduce:
            window_size_s = 600.0  # 10 minutes
            hop_size_s = 480.0     # 8 minutes (2-minute overlap)
            print(f"[hook_finder] Long lecture detected ({total_duration:.1f}s > {self.map_reduce_threshold:.0f}s). "
                  f"Executing Map-Reduce in 10-minute sliding windows (2-minute overlap)...")

            win_idx = 0
            cur_start = 0.0
            while cur_start < total_duration:
                cur_end = min(total_duration, cur_start + window_size_s)
                # Filter segments falling into this window
                win_lines = [
                    annotated_lines[i] for i, s in enumerate(segments)
                    if s["end"] >= cur_start and s["start"] <= cur_end
                ]

                if win_lines:
                    win_idx += 1
                    print(f"[hook_finder] Map window #{win_idx}: {format_seconds_to_timestamp(cur_start)} -> "
                          f"{format_seconds_to_timestamp(cur_end)} ({len(win_lines)} segments)...")
                    win_transcript = "\n".join(win_lines)
                    cands = self._call_gemini_candidates(
                        client=client,
                        formatted_transcript=win_transcript,
                        num_clips=max(2, num_clips)
                    )
                    raw_candidates.extend(cands)

                cur_start += hop_size_s
                if cur_end >= total_duration:
                    break
        else:
            print(f"[hook_finder] Prompting {self.llm_model_name} to detect top {num_clips} viral hooks (single window)...")
            full_transcript_text = "\n".join(annotated_lines)
            raw_candidates = self._call_gemini_candidates(
                client=client,
                formatted_transcript=full_transcript_text,
                num_clips=num_clips
            )

        elapsed = time.time() - t0
        print(f"[hook_finder] AI Hook detection finished in {elapsed:.2f}s ({len(raw_candidates)} proposals)")

        # 3. Convert segment IDs to concrete seconds and compute weighted scores
        converted_clips: List[ViralClipMetadata] = []
        for cand in raw_candidates:
            meta = self._convert_candidate_to_metadata(
                item=cand,
                segment_map=segment_map,
                total_duration=total_duration,
                client=client
            )
            if meta is not None:
                converted_clips.append(meta)

        # 4. Deduplicate overlapping clips across windows
        deduped_clips = deduplicate_candidate_clips(converted_clips)
        print(f"[hook_finder] Candidate pool after deduplication: {len(deduped_clips)} clips")

        # 5. Optional Face Presence Filter
        if self.filter_face_presence and v_path.is_file():
            face_filtered_clips: List[ViralClipMetadata] = []
            for c in deduped_clips:
                if c.aspect_ratio == "9:16":
                    passes_face, face_ratio = check_face_presence(v_path, c.start_time, c.end_time, min_face_ratio=0.70)
                    if not passes_face:
                        print(f"[hook_finder] Dropping vertical candidate '{c.title}' (teacher face visible in only {face_ratio * 100:.1f}% < 70% frames)")
                        continue
                face_filtered_clips.append(c)
            deduped_clips = face_filtered_clips

        # 6. Global Rerank if candidates exceed requested count
        if len(deduped_clips) > num_clips:
            final_clips = self._rerank_candidates_gemini(client, deduped_clips, num_clips)
        else:
            final_clips = sorted(deduped_clips, key=lambda c: c.viral_score, reverse=True)[:num_clips]

        # 7. Renumber sequential IDs
        for idx, c in enumerate(final_clips, 1):
            c.clip_id = idx

        return final_clips


def cut_snippet_without_reencode(
    src_video: Path | str,
    out_snippet: Path | str,
    start_time: float,
    end_time: float,
    target_fps: Optional[float] = None,
    audio_delay_ms: float = 500.0,
    crf: int = 10,
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
        audio_delay_ms=audio_delay_ms,
        crf=crf,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Highlight Detection and Hook Finder for lectures.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--clips", type=int, default=3, help="Number of viral clips to detect")
    parser.add_argument("--whisper-model", default="base", help="Whisper model for full lecture pass")
    parser.add_argument("--filter-face-presence", action="store_true", help="Filter vertical clips with < 70%% presenter visibility")
    args = parser.parse_args()

    finder = HookFinder(
        whisper_model=args.whisper_model,
        filter_face_presence=args.filter_face_presence
    )
    clips = finder.find_viral_clips(args.video, num_clips=args.clips)
    print(f"\n[Detected {len(clips)} Viral Clips]:")
    for c in clips:
        print(f"\n#{c.clip_id}: {c.title} (Score: {c.viral_score}/100)")
        print(f"  Time: {format_seconds_to_timestamp(c.start_time)} -> {format_seconds_to_timestamp(c.end_time)} ({c.duration:.1f}s)")
        print(f"  Hook: \"{c.hook}\"")
        print(f"  Why : {c.reason}")
