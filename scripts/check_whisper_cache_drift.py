#!/usr/bin/env python3
"""
check_whisper_cache_drift.py - Verification script comparing cached-slice words
against a fresh Whisper pass on a cut snippet.

Mathematical Time Mapping Invariant:
  final_t = orig_t - clip_start + audio_delay_s
  (audio_delay_s = audio_delay_ms / 1000, because the final audio is delayed).

Requirements:
- Slices word timestamps from full-lecture disk cache.
- Cuts snippet using cut_snippet_accurate with audio_delay_ms.
- Runs fresh Whisper pass on snippet.
- Aligns words and calculates drift (target <= 0.15s).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.av_sync import cut_snippet_accurate
from core.hook_finder import HookFinder
from core.transcriber import DEFAULT_WHISPER_INITIAL_PROMPT
from core.whisper_backend import (
    Word,
    load_lecture_transcript_cache,
    slice_cached_words_for_clip,
    transcribe_words,
)


def align_and_measure_drift(
    sliced_words: List[Word],
    fresh_words: List[Word],
    max_time_diff: float = 0.35,
) -> Tuple[List[dict], float, float, float, float]:
    """
    Monotonically match identical or near-identical words between sliced and fresh passes.
    Returns:
        (matched_records, max_start_drift, max_end_drift, avg_start_drift, avg_end_drift)
    """
    matched: List[dict] = []
    last_fresh_idx = 0

    norm_sliced = [
        (w.text.strip().lower().strip(".,!?।॥\"'"), round(w.start, 3), round(w.end, 3), w.text)
        for w in sliced_words
    ]
    norm_fresh = [
        (w.text.strip().lower().strip(".,!?।॥\"'"), round(w.start, 3), round(w.end, 3), w.text)
        for w in fresh_words
    ]

    for s_clean, s_st, s_et, s_raw in norm_sliced:
        if not s_clean:
            continue
        # Search ahead in fresh words within a localized window
        search_window = min(len(norm_fresh), last_fresh_idx + 6)
        for j in range(last_fresh_idx, search_window):
            f_clean, f_st, f_et, f_raw = norm_fresh[j]
            if f_clean == s_clean and abs(s_st - f_st) <= max_time_diff:
                st_drift = abs(s_st - f_st)
                et_drift = abs(s_et - f_et)
                matched.append({
                    "word": s_raw,
                    "sliced_start": s_st,
                    "fresh_start": f_st,
                    "start_drift": st_drift,
                    "sliced_end": s_et,
                    "fresh_end": f_et,
                    "end_drift": et_drift,
                })
                last_fresh_idx = j + 1
                break

    if not matched:
        return [], 999.0, 999.0, 999.0, 999.0

    max_st_drift = max(m["start_drift"] for m in matched)
    max_et_drift = max(m["end_drift"] for m in matched)
    avg_st_drift = sum(m["start_drift"] for m in matched) / len(matched)
    avg_et_drift = sum(m["end_drift"] for m in matched) / len(matched)

    return matched, max_st_drift, max_et_drift, avg_st_drift, avg_et_drift


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify timestamp drift between cached-slice words and fresh Whisper pass on sample."
    )
    parser.add_argument(
        "--input", "-i",
        default="samples/sample.mp4",
        help="Path to input sample video (default: samples/sample.mp4)"
    )
    parser.add_argument(
        "--clip-start",
        type=float,
        default=5.0,
        help="Clip start time in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--clip-end",
        type=float,
        default=25.0,
        help="Clip end time in seconds (default: 25.0)"
    )
    parser.add_argument(
        "--audio-delay",
        type=float,
        default=500.0,
        help="Audio delay in milliseconds (default: 500.0)"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model (default: base)"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "openai", "mlx"],
        help="Whisper backend (default: auto)"
    )
    parser.add_argument(
        "--max-drift-target",
        type=float,
        default=0.15,
        help="Target maximum drift tolerance in seconds (default: 0.15s)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"[ERROR] Sample video not found: {input_path}")

    clip_start = float(args.clip_start)
    clip_end = float(args.clip_end)
    audio_delay_ms = float(args.audio_delay)
    model = args.model
    backend = args.backend
    max_drift_target = float(args.max_drift_target)

    print("=" * 70)
    print(" WHISPER CACHE-SLICING VS FRESH PASS DRIFT VERIFICATION")
    print("=" * 70)
    print(f"• Input Video    : {input_path.name}")
    print(f"• Slice Interval : {clip_start:.2f}s -> {clip_end:.2f}s ({clip_end - clip_start:.1f}s)")
    print(f"• Audio Delay    : {audio_delay_ms:.0f}ms (offset = {audio_delay_ms/1000.0:.3f}s)")
    print(f"• Whisper Model  : {model} (backend={backend})")
    print(f"• Drift Target   : <= {max_drift_target:.2f}s")
    print("=" * 70)

    # Step 1: Ensure full-lecture transcript is cached
    print("\n[1/4] Ensuring full-lecture transcript cache exists with word-level timestamps...")
    finder = HookFinder(whisper_model=model, backend=backend)
    finder.get_full_transcript(input_path)

    cached_data = load_lecture_transcript_cache(input_path, model_name=model)
    if not cached_data or not cached_data.get("words"):
        sys.exit("[ERROR] Failed to load cached words for lecture.")

    cached_words = cached_data["words"]
    cached_segments = cached_data.get("segments", [])
    print(f"[1/4] Cache loaded successfully: {len(cached_words)} total words in video.")

    # Step 2: Slice cached words using exact time mapping formula
    print(f"\n[2/4] Slicing words for [{clip_start:.2f}s -> {clip_end:.2f}s] via time mapping:")
    print("      final_t = orig_t - clip_start + audio_delay_s ...")
    sliced_words = slice_cached_words_for_clip(
        cached_words=cached_words,
        clip_start=clip_start,
        clip_end=clip_end,
        audio_delay_ms=audio_delay_ms,
        cached_segments=cached_segments,
        min_coverage=0.80,
    )
    if not sliced_words:
        sys.exit("[ERROR] Slicing returned None (word coverage < 80%). Check input timestamps.")

    # Exclude partial boundary words that started before the cut boundary (mid-phoneme truncation)
    audio_delay_s = audio_delay_ms / 1000.0
    complete_sliced = [w for w in sliced_words if w.start >= audio_delay_s]
    print(f"[2/4] Sliced {len(sliced_words)} words ({len(complete_sliced)} fully inside clip window).")

    # Step 3: Cut accurate snippet and perform fresh Whisper pass
    print(f"\n[3/4] Cutting snippet with accurate audio delay ({audio_delay_ms:.0f}ms) and transcribing...")
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_snippet:
        cut_snippet_accurate(
            src_video=input_path,
            out_snippet=tmp_snippet.name,
            start_time=clip_start,
            end_time=clip_end,
            audio_delay_ms=audio_delay_ms,
        )

        fresh_res = transcribe_words(
            audio_path=tmp_snippet.name,
            model=model,
            backend=backend,
            initial_prompt=DEFAULT_WHISPER_INITIAL_PROMPT,
            word_timestamps=True,
        )

    fresh_words = fresh_res.words
    print(f"[3/4] Fresh pass produced {len(fresh_words)} words on cut snippet.")

    # Step 4: Align words and measure timestamp drift
    print("\n[4/4] Aligning words and computing drift metrics...")
    matched, max_st, max_et, avg_st, avg_et = align_and_measure_drift(
        sliced_words=complete_sliced,
        fresh_words=fresh_words,
        max_time_diff=0.35,
    )

    if not matched:
        sys.exit("[ERROR] Could not align any words between sliced cache and fresh snippet.")

    print("\n" + "-" * 75)
    print(f" {'Word':<16} | {'Sliced Interval':<17} | {'Fresh Interval':<17} | {'Start Drift':>11}")
    print("-" * 75)
    for m in matched[:15]:
        s_span = f"{m['sliced_start']:05.2f}s -> {m['sliced_end']:05.2f}s"
        f_span = f"{m['fresh_start']:05.2f}s -> {m['fresh_end']:05.2f}s"
        print(f" {m['word']:<16} | {s_span:<17} | {f_span:<17} | {m['start_drift']:>10.4f}s")
    if len(matched) > 15:
        print(f" ... and {len(matched) - 15} more aligned words.")
    print("-" * 75)

    print("\n" + "=" * 70)
    print(" DRIFT MEASUREMENT SUMMARY:")
    print("=" * 70)
    print(f"• Aligned Words  : {len(matched)} / {min(len(sliced_words), len(fresh_words))}")
    print(f"• Max Start Drift: {max_st:.4f}s  (Target <= {max_drift_target:.2f}s) [{'PASS' if max_st <= max_drift_target else 'FAIL'}]")
    print(f"• Max End Drift  : {max_et:.4f}s  (Target <= {max_drift_target:.2f}s) [{'PASS' if max_et <= max_drift_target else 'FAIL'}]")
    print(f"• Mean Start Drift: {avg_st:.4f}s")
    print(f"• Mean End Drift : {avg_et:.4f}s")
    print("=" * 70)

    if max_st <= max_drift_target and max_et <= max_drift_target:
        print(f"\n>>> VERIFICATION SUCCESSFUL: Maximum drift ({max(max_st, max_et):.4f}s) <= target ({max_drift_target:.2f}s)!")
        sys.exit(0)
    else:
        print(f"\n[WARNING] Maximum drift ({max(max_st, max_et):.4f}s) exceeded target ({max_drift_target:.2f}s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
