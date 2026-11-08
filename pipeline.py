#!/usr/bin/env python3
"""
pipeline.py - Unified High-Performance CLI Pipeline for Long Lectures to 9:16 Viral Reels.

Hardware Target: Apple Silicon (MacBook Air M4, 16GB RAM, macOS, Python 3.14).

Subcommands:
  python3 pipeline.py run --input lecture.mp4 --clips 3      # End-to-end: AI Hook Detection + 9:16 Reframe + Hinglish Captions
  python3 pipeline.py detect --input lecture.mp4 --clips 5   # Only detect top viral hooks & print summary
  python3 pipeline.py clip --input lecture.mp4 --start 15.0 --end 48.0  # Process a specific timestamp slice
  python3 pipeline.py reframe --input video.mp4 -o out.mp4   # Standalone face-tracking 9:16 crop
  python3 pipeline.py caption --input vertical.mp4 -o out.mp4 # Standalone Hinglish word-by-word caption burning
  python3 pipeline.py test                                   # System diagnostics and health check
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.boundary_refiner import refine_clip_boundaries
from core.captioner import CaptionRenderer, list_styles
from core.health_check import run_diagnostics
from core.hook_finder import (
    HookFinder,
    ViralClipMetadata,
    cut_snippet_without_reencode,
    format_seconds_to_timestamp,
)
from core.reframer import FaceReframer, ReframeConfig, ReframePlan, reframe_video
from core.render import render_reel
from core.transcriber import HinglishTranscriber


def print_profile_table(stages: Dict[str, float], clip_title: str) -> None:
    """Print an ASCII table of wall-clock time and percentage share per pipeline stage."""
    total_time = sum(stages.values())
    w_stage = max(max(len(k) for k in stages.keys()), 18)
    print("\n" + "=" * 65)
    print(f" PIPELINE STAGE TIMING PROFILE: {clip_title}")
    print("=" * 65)
    print(f"{'Stage':<{w_stage}} | {'Time (s)':>10} | {'Share (%)':>10}")
    print("-" * (w_stage + 25))
    for stage, elapsed in stages.items():
        share = (elapsed / total_time * 100.0) if total_time > 0 else 0.0
        print(f"{stage:<{w_stage}} | {elapsed:>9.2f}s | {share:>9.1f}%")
    print("-" * (w_stage + 25))
    print(f"{'Total Stage Time':<{w_stage}} | {total_time:>9.2f}s | {'100.0%':>10}")
    print("=" * 65 + "\n")


def save_profile_json(
    output_dir: Path,
    out_reel_path: Path,
    clip_info: ViralClipMetadata,
    stage_timings: Dict[str, float],
    total_wall_time: float
) -> Path:
    """Write stage timings to output/profile_<clip>.json."""
    total_stage_time = round(sum(stage_timings.values()), 3)
    profile_data = {
        "clip": out_reel_path.stem,
        "clip_id": clip_info.clip_id,
        "slug": clip_info.slug,
        "title": clip_info.title,
        "duration_s": clip_info.duration,
        "output_file": str(out_reel_path),
        "stages": stage_timings,
        "total_stage_time_s": total_stage_time,
        "total_wall_time_s": round(total_wall_time, 3),
        **stage_timings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary profile json path: output/profile_<clip>.json
    profile_json_path = output_dir / f"profile_{out_reel_path.stem}.json"
    profile_json_path.write_text(json.dumps(profile_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also ensure profile_<clip_id>_<slug>.json exists if stem differs
    alt_name = f"profile_{clip_info.clip_id:02d}_{clip_info.slug}.json"
    alt_path = output_dir / alt_name
    if alt_path != profile_json_path:
        alt_path.write_text(json.dumps(profile_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[profile] Wrote stage timings to: {profile_json_path}")
    return profile_json_path


@dataclass
class PreparedClip:
    """Pre-computed assets, plans, and metadata ready for rendering."""
    clip_info: ViralClipMetadata
    snippet_path: Path
    reframed_path: Path
    temp_dir: Optional[tempfile.TemporaryDirectory]
    plan: Optional[ReframePlan]
    words: List
    time_cut: float
    time_boundary_refine: float
    time_reframe_p1: float
    time_reframe_render: float
    time_whisper: float
    time_romanize: float
    is_widescreen: bool
    is_4_5: bool
    layout: str
    target_aspect: str
    chosen_style: str
    effective_chunk_size: int
    effective_bottom_offset: int
    out_reel_path: Path
    t_clip_overall_start: float


def prepare_clip(
    src_video: Path,
    clip_info: ViralClipMetadata,
    output_dir: Path,
    output_path: Optional[Path] = None,
    whisper_clip_model: str = "large-v3",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    smooth_alpha: float = 0.06,
    chunk_size: int = 4,
    bottom_offset: int = 380,
    save_intermediate: bool = False,
    no_llm: bool = False,
    style: str = "classic-yellow",
    reframe_preset: str = "talking_head",
    layout: str = "direct-crop",
    audio_delay_ms: float = 500.0,
    aspect: str = "auto",
    no_caption: bool = False,
    refine_boundaries: bool = False,
    whisper_backend: str = "auto",
    legacy_render: bool = False,
    detect_every: int = 3,
    punch_in: str = "off",
    detector: str = "haar",
    model_path: Optional[Path | str] = None,
) -> PreparedClip:
    """
    Prepare clip N: snippet cut, reframing plan, word timestamps, and romanization.
    Can be run in a background worker thread while clip N-1 is rendering.
    """
    t_clip_overall_start = time.time()
    time_boundary_refine = 0.0
    time_cut = 0.0
    time_reframe_p1 = 0.0
    time_reframe_render = 0.0
    time_whisper = 0.0
    time_romanize = 0.0

    # 0. High-Precision Boundary Re-verification if requested
    if refine_boundaries:
        t_br_start = time.time()
        refined_start, refined_end = refine_clip_boundaries(
            video_path=src_video,
            start_t=clip_info.start_time,
            end_t=clip_info.end_time,
            margin_s=60.0,
            backend=whisper_backend,
        )
        time_boundary_refine = time.time() - t_br_start
        clip_info.start_time = refined_start
        clip_info.end_time = refined_end
        clip_info.duration = round(refined_end - refined_start, 2)

    # 1. Resolve target aspect ratio branching
    target_aspect = aspect.lower().replace("x", ":")
    if target_aspect == "auto":
        target_aspect = getattr(clip_info, "aspect_ratio", None) or ("16:9" if clip_info.duration >= 180.0 else "9:16")
    if target_aspect not in ("9:16", "16:9", "4:5"):
        target_aspect = "9:16" if clip_info.duration < 180.0 else "16:9"

    is_widescreen = (target_aspect == "16:9")
    is_4_5 = (target_aspect == "4:5")

    # Output filename naming convention
    if output_path:
        out_reel_path = output_path
    else:
        if is_widescreen:
            prefix = f"clip_16x9_{clip_info.clip_id:02d}_{clip_info.slug}.mp4"
        elif is_4_5:
            prefix = f"feed_4x5_{clip_info.clip_id:02d}_{clip_info.slug}.mp4"
        elif layout == "blur-bw":
            prefix = f"reel_blur_bw_{clip_info.clip_id:02d}_{clip_info.slug}.mp4"
        else:
            prefix = f"reel_{clip_info.clip_id:02d}_{clip_info.slug}.mp4"
        out_reel_path = output_dir / prefix

    if is_widescreen:
        label = "16:9 WIDESCREEN CLIP"
    elif is_4_5:
        label = "4:5 INSTAGRAM FEED CLIP"
    elif layout == "blur-bw":
        label = "1:1 BLUR-BW AESTHETIC REEL"
    else:
        label = "9:16 VIRAL REEL"

    render_mode_desc = "Legacy Multi-Step (--legacy-render)" if legacy_render else "Unified Single-Pass (render_reel)"
    print("\n" + "-" * 65)
    print(f"PREPARING {label} #{clip_info.clip_id}: \"{clip_info.title}\"")
    print(f"Duration: {clip_info.duration:.1f}s ({format_seconds_to_timestamp(clip_info.start_time)} -> {format_seconds_to_timestamp(clip_info.end_time)})")
    print(f"Aspect Ratio: {target_aspect} | Canvas Layout: {layout} | Render Mode: {render_mode_desc}")
    print(f"Captions: {'Disabled (--no-caption)' if no_caption else f'Enabled ({style})'}")
    print(f"Hook: \"{clip_info.hook}\"")
    print("-" * 65)

    if save_intermediate:
        inter_dir = output_dir / "intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{target_aspect.replace(':', 'x')}_{clip_info.clip_id:02d}_{clip_info.slug}"
        snippet_path = inter_dir / f"snippet_{tag}.mp4"
        reframed_path = inter_dir / f"reframed_{tag}.mp4"
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"clip_{clip_info.clip_id}_")
        snippet_path = Path(temp_dir.name) / "snippet.mp4"
        reframed_path = Path(temp_dir.name) / "reframed.mp4"

    # Step A: Cut snippet with accurate timestamp zeroing & audio delay lip-sync (high quality CRF<=12)
    print(f"\n[Clip {clip_info.clip_id}] [1/4] Cutting snippet from {clip_info.start_time:.1f}s to {clip_info.end_time:.1f}s (audio_delay={audio_delay_ms:.0f}ms, crf=10)...")
    t0 = time.time()
    cut_snippet_without_reencode(
        src_video,
        snippet_path,
        clip_info.start_time,
        clip_info.end_time,
        audio_delay_ms=audio_delay_ms,
        crf=10,
    )
    time_cut = time.time() - t0
    print(f"[Clip {clip_info.clip_id}] Snippet cut in {time_cut:.2f}s")

    plan: Optional[ReframePlan] = None
    words = []

    if legacy_render:
        # Step B: Reframe to intermediate video
        if is_widescreen:
            print(f"\n[Clip {clip_info.clip_id}] [2/4] Preserving native 16:9 widescreen format (bypassing FaceReframer)...")
            active_video_path = snippet_path
            time_reframe_p1 = 0.0
            time_reframe_render = 0.0
        else:
            reframe_label = "4:5 portrait" if is_4_5 else ("1:1 blur-bw" if layout == "blur-bw" else "9:16 vertical")
            print(f"\n[Clip {clip_info.clip_id}] [2/4] Face-tracking {reframe_label} reframing (layout={layout}, aspect={target_aspect}, punch_in={punch_in})...")
            reframe_cfg = ReframeConfig(
                src=snippet_path,
                out=reframed_path,
                smooth_alpha=smooth_alpha,
                preset=reframe_preset,
                layout=layout,
                audio_delay_ms=0.0,
                aspect_ratio=target_aspect,
                detect_every=detect_every,
                punch_in=punch_in,
                detector=detector,
                model_path=model_path,
            )
            reframer = FaceReframer(reframe_cfg)
            reframer.run()
            active_video_path = reframed_path
            time_reframe_p1 = getattr(reframer, "time_pass1", 0.0)
            time_reframe_render = getattr(reframer, "time_render", 0.0)

        # Step C: Word-level Whisper transcription & Gemini Hinglish romanization
        if not no_caption:
            print(f"\n[Clip {clip_info.clip_id}] [3/4] Word-level Whisper STT & Hinglish transliteration...")
            transcriber = HinglishTranscriber(
                whisper_model=whisper_clip_model,
                llm_model=llm_model,
                api_key=api_key,
                no_llm=no_llm,
                uppercase=True,
                backend=whisper_backend,
            )
            words = transcriber.transcribe_and_romanize(
                video_path=active_video_path,
                src_lecture_video=src_video,
                clip_start=clip_info.start_time,
                clip_end=clip_info.end_time,
                audio_delay_ms=audio_delay_ms,
            )
            time_whisper = getattr(transcriber, "time_whisper", 0.0)
            time_romanize = getattr(transcriber, "time_romanize", 0.0)
    else:
        # Step B: Word-level Whisper transcription & Gemini Hinglish romanization
        if not no_caption:
            print(f"\n[Clip {clip_info.clip_id}] [2/4] Word-level Whisper STT & Hinglish transliteration...")
            transcriber = HinglishTranscriber(
                whisper_model=whisper_clip_model,
                llm_model=llm_model,
                api_key=api_key,
                no_llm=no_llm,
                uppercase=True,
                backend=whisper_backend,
            )
            words = transcriber.transcribe_and_romanize(
                video_path=snippet_path,
                src_lecture_video=src_video,
                clip_start=clip_info.start_time,
                clip_end=clip_info.end_time,
                audio_delay_ms=audio_delay_ms,
            )
            time_whisper = getattr(transcriber, "time_whisper", 0.0)
            time_romanize = getattr(transcriber, "time_romanize", 0.0)
        else:
            print(f"\n[Clip {clip_info.clip_id}] [2/4] Captions disabled (--no-caption). Bypassing Whisper & transliteration...")

        # Step C: Compute reframing plan (trajectory, geometry, scene cuts)
        if is_widescreen:
            print(f"\n[Clip {clip_info.clip_id}] [3/4] Preserving native 16:9 widescreen format (bypassing FaceReframer)...")
            time_reframe_p1 = 0.0
        else:
            reframe_label = "4:5 portrait" if is_4_5 else ("1:1 blur-bw" if layout == "blur-bw" else "9:16 vertical")
            print(f"\n[Clip {clip_info.clip_id}] [3/4] Planning face-tracking {reframe_label} reframing (layout={layout}, aspect={target_aspect}, punch_in={punch_in})...")
            reframe_cfg = ReframeConfig(
                src=snippet_path,
                out=reframed_path,
                smooth_alpha=smooth_alpha,
                preset=reframe_preset,
                layout=layout,
                audio_delay_ms=0.0,
                aspect_ratio=target_aspect,
                detect_every=detect_every,
                punch_in=punch_in,
                words=words,
                detector=detector,
                model_path=model_path,
            )
            reframer = FaceReframer(reframe_cfg)
            plan = reframer.plan(words=words)
            time_reframe_p1 = getattr(reframer, "time_pass1", 0.0)
            if save_intermediate:
                reframer.render(plan)

    chosen_style = style
    effective_chunk_size = chunk_size
    effective_bottom_offset = bottom_offset

    if is_widescreen:
        if style == "classic-yellow":
            chosen_style = "subtitle-16x9"
        if chunk_size == 4:
            effective_chunk_size = 6
        if bottom_offset == 380:
            effective_bottom_offset = 100

    return PreparedClip(
        clip_info=clip_info,
        snippet_path=snippet_path,
        reframed_path=reframed_path,
        temp_dir=temp_dir,
        plan=plan,
        words=words,
        time_cut=time_cut,
        time_boundary_refine=time_boundary_refine,
        time_reframe_p1=time_reframe_p1,
        time_reframe_render=time_reframe_render,
        time_whisper=time_whisper,
        time_romanize=time_romanize,
        is_widescreen=is_widescreen,
        is_4_5=is_4_5,
        layout=layout,
        target_aspect=target_aspect,
        chosen_style=chosen_style,
        effective_chunk_size=effective_chunk_size,
        effective_bottom_offset=effective_bottom_offset,
        out_reel_path=out_reel_path,
        t_clip_overall_start=t_clip_overall_start,
    )


def render_prepared_clip(
    prepared: PreparedClip,
    output_dir: Path,
    profile: bool = False,
    legacy_render: bool = False,
    fast: bool = False,
    no_caption: bool = False,
) -> Path:
    """Render a pre-prepared clip to its final output file."""
    try:
        clip_info = prepared.clip_info
        out_reel_path = prepared.out_reel_path
        time_caption_burn = 0.0
        time_unified_render = 0.0

        if legacy_render:
            active_video_path = prepared.reframed_path if not prepared.is_widescreen else prepared.snippet_path
            if no_caption:
                print(f"\n[Clip {clip_info.clip_id}] [4/4] Captions disabled (--no-caption). Saving clean video...")
                shutil.copy2(active_video_path, out_reel_path)
            else:
                print(f"\n[Clip {clip_info.clip_id}] [4/4] Burning animated captions to final video...")
                renderer = CaptionRenderer(
                    chunk_size=prepared.effective_chunk_size,
                    bottom_offset=prepared.effective_bottom_offset,
                    style=prepared.chosen_style,
                )
                t_burn_start = time.time()
                renderer.render_to_video(
                    input_video=active_video_path,
                    output_video=out_reel_path,
                    words=prepared.words,
                    fast=fast,
                )
                time_caption_burn = time.time() - t_burn_start

            print(f"\n>>> {'CLIP' if prepared.is_widescreen else 'REEL'} #{clip_info.clip_id} READY: {out_reel_path.name}")
            if profile:
                stage_timings: Dict[str, float] = {
                    "cut": round(prepared.time_cut, 3),
                    "boundary refine": round(prepared.time_boundary_refine, 3),
                    "reframe pass 1": round(prepared.time_reframe_p1, 3),
                    "reframe render": round(prepared.time_reframe_render, 3),
                    "whisper": round(prepared.time_whisper, 3),
                    "gemini romanize": round(prepared.time_romanize, 3),
                    "caption burn": round(time_caption_burn, 3),
                }
                clip_title = f"Clip #{clip_info.clip_id:02d} ({clip_info.slug})"
                print_profile_table(stage_timings, clip_title=clip_title)
                save_profile_json(
                    output_dir=output_dir,
                    out_reel_path=out_reel_path,
                    clip_info=clip_info,
                    stage_timings=stage_timings,
                    total_wall_time=time.time() - prepared.t_clip_overall_start
                )
            return out_reel_path

        else:
            # Modern Single-Pass Render Path (render_reel)
            print(f"\n[Clip {clip_info.clip_id}] [4/4] Single-pass unified video encoding (crop/scale + captions -> final video)...")
            t_render_start = time.time()
            render_reel(
                snippet=prepared.snippet_path,
                plan=prepared.plan,
                words=prepared.words if not no_caption else None,
                style=prepared.chosen_style,
                out_path=out_reel_path,
                chunk_size=prepared.effective_chunk_size,
                bottom_offset=prepared.effective_bottom_offset,
                no_caption=no_caption,
                fast=fast,
            )
            time_unified_render = time.time() - t_render_start

            print(f"\n>>> {'CLIP' if prepared.is_widescreen else 'REEL'} #{clip_info.clip_id} READY: {out_reel_path.name}")
            if profile:
                stage_timings = {
                    "cut": round(prepared.time_cut, 3),
                    "boundary refine": round(prepared.time_boundary_refine, 3),
                    "reframe pass 1": round(prepared.time_reframe_p1, 3),
                    "reframe render": 0.0,
                    "whisper": round(prepared.time_whisper, 3),
                    "gemini romanize": round(prepared.time_romanize, 3),
                    "caption burn": 0.0,
                    "unified render": round(time_unified_render, 3),
                }
                clip_title = f"Clip #{clip_info.clip_id:02d} ({clip_info.slug})"
                print_profile_table(stage_timings, clip_title=clip_title)
                save_profile_json(
                    output_dir=output_dir,
                    out_reel_path=out_reel_path,
                    clip_info=clip_info,
                    stage_timings=stage_timings,
                    total_wall_time=time.time() - prepared.t_clip_overall_start
                )
            return out_reel_path
    finally:
        if prepared.temp_dir is not None:
            prepared.temp_dir.cleanup()


def process_single_clip(
    src_video: Path,
    clip_info: ViralClipMetadata,
    output_dir: Path,
    output_path: Optional[Path] = None,
    whisper_clip_model: str = "large-v3",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    smooth_alpha: float = 0.06,
    chunk_size: int = 4,
    bottom_offset: int = 380,
    save_intermediate: bool = False,
    no_llm: bool = False,
    style: str = "classic-yellow",
    reframe_preset: str = "talking_head",
    layout: str = "direct-crop",
    audio_delay_ms: float = 500.0,
    aspect: str = "auto",
    no_caption: bool = False,
    refine_boundaries: bool = False,
    profile: bool = False,
    whisper_backend: str = "auto",
    legacy_render: bool = False,
    fast: bool = False,
    detect_every: int = 3,
    punch_in: str = "off",
    detector: str = "haar",
    model_path: Optional[Path | str] = None,
) -> Path:
    """Process a single clip through prepare + render."""
    prepared = prepare_clip(
        src_video=src_video,
        clip_info=clip_info,
        output_dir=output_dir,
        output_path=output_path,
        whisper_clip_model=whisper_clip_model,
        llm_model=llm_model,
        api_key=api_key,
        smooth_alpha=smooth_alpha,
        chunk_size=chunk_size,
        bottom_offset=bottom_offset,
        save_intermediate=save_intermediate,
        no_llm=no_llm,
        style=style,
        reframe_preset=reframe_preset,
        layout=layout,
        audio_delay_ms=audio_delay_ms,
        aspect=aspect,
        no_caption=no_caption,
        refine_boundaries=refine_boundaries,
        whisper_backend=whisper_backend,
        legacy_render=legacy_render,
        detect_every=detect_every,
        punch_in=punch_in,
        detector=detector,
        model_path=model_path,
    )
    return render_prepared_clip(
        prepared=prepared,
        output_dir=output_dir,
        profile=profile,
        legacy_render=legacy_render,
        fast=fast,
        no_caption=no_caption,
    )


def process_clips_pipelined(
    clips: List[ViralClipMetadata],
    src_video: Path,
    output_dir: Path,
    whisper_clip_model: str = "large-v3",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    smooth_alpha: float = 0.06,
    chunk_size: int = 4,
    bottom_offset: int = 380,
    save_intermediate: bool = False,
    no_llm: bool = False,
    style: str = "classic-yellow",
    reframe_preset: str = "talking_head",
    layout: str = "direct-crop",
    audio_delay_ms: float = 500.0,
    aspect: str = "auto",
    no_caption: bool = False,
    refine_boundaries: bool = False,
    profile: bool = False,
    whisper_backend: str = "auto",
    legacy_render: bool = False,
    fast: bool = False,
    detect_every: int = 3,
    punch_in: str = "off",
    detector: str = "haar",
    model_path: Optional[Path | str] = None,
) -> Tuple[List[Path], List[dict]]:
    """
    Process multiple clips with pipelined preparation:
    Clip N+1 is prepared in a background ThreadPoolExecutor(max_workers=1)
    while clip N is being rendered. Respects Gemini API rate limits with
    sequential preparation and persistent caching.
    """
    generated_reels: List[Path] = []
    metadata_list: List[dict] = []

    if not clips:
        return generated_reels, metadata_list

    def _do_prepare(clip: ViralClipMetadata) -> PreparedClip:
        return prepare_clip(
            src_video=src_video,
            clip_info=clip,
            output_dir=output_dir,
            whisper_clip_model=whisper_clip_model,
            llm_model=llm_model,
            api_key=api_key,
            smooth_alpha=smooth_alpha,
            chunk_size=chunk_size,
            bottom_offset=bottom_offset,
            save_intermediate=save_intermediate,
            no_llm=no_llm,
            style=style,
            reframe_preset=reframe_preset,
            layout=layout,
            audio_delay_ms=audio_delay_ms,
            aspect=aspect,
            no_caption=no_caption,
            refine_boundaries=refine_boundaries,
            whisper_backend=whisper_backend,
            legacy_render=legacy_render,
            detect_every=detect_every,
            punch_in=punch_in,
            detector=detector,
            model_path=model_path,
        )

    # Use ThreadPoolExecutor(max_workers=1) to pipeline preparation of clip N+1 while clip N renders
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_next = executor.submit(_do_prepare, clips[0])
        for idx in range(len(clips)):
            prepared = future_next.result()

            # Start background preparation for clip N+1 while clip N renders
            if idx + 1 < len(clips):
                future_next = executor.submit(_do_prepare, clips[idx + 1])
            else:
                future_next = None

            reel_file = render_prepared_clip(
                prepared=prepared,
                output_dir=output_dir,
                profile=profile,
                legacy_render=legacy_render,
                fast=fast,
                no_caption=no_caption,
            )
            generated_reels.append(reel_file)

            meta_dict = clips[idx].model_dump()
            meta_dict["output_file"] = str(reel_file)
            meta_dict["filename"] = reel_file.name
            metadata_list.append(meta_dict)

    return generated_reels, metadata_list


def handle_run(args: argparse.Namespace) -> None:
    """Execute end-to-end automated pipeline."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"[ERROR] Input video not found: {input_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" LECTURE-TO-REEL AI AUTOMATED PIPELINE (Apple Silicon M4)")
    print("=" * 70)
    print(f"• Input Video     : {input_path.name}")
    print(f"• Target Clips    : {args.clips} clips")
    print(f"• Target Aspect   : {args.aspect} (<180s -> 9:16 vertical, >=180s -> 16:9 widescreen)")
    print(f"• Captions        : {'Disabled (--no-caption)' if args.no_caption else f'Enabled ({args.style})'}")
    print(f"• Whisper Backend : {getattr(args, 'whisper_backend', 'auto')}")
    print(f"• Face Detector   : {getattr(args, 'detector', 'haar')}")
    print(f"• Full Scan Model : Whisper '{args.whisper_full_model}' (Cached)")
    print(f"• Word Sync Model : Whisper '{args.whisper_clip_model}'")
    print(f"• AI Brain        : {args.llm_model}")
    print(f"• Reframe Preset  : {args.reframe_preset}")
    print(f"• Video Layout    : {args.layout}")
    print(f"• Audio Delay     : {args.audio_delay:.0f}ms (Lip-sync alignment)")
    print(f"• Caption Style   : {args.style}")
    print(f"• Output Directory: {output_dir}")
    print("=" * 70)

    # 1. AI Highlight & Hook Detection
    t_start = time.time()
    finder = HookFinder(
        whisper_model=args.whisper_full_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        backend=getattr(args, "whisper_backend", "auto"),
        filter_face_presence=getattr(args, "filter_face_presence", False),
    )
    clips = finder.find_viral_clips(input_path, num_clips=args.clips)

    if not clips:
        sys.exit("[ERROR] No suitable highlight clips found in lecture transcript.")

    print("\n" + "=" * 70)
    print(" DETECTED VIRAL HIGHLIGHT HOOKS:")
    print("=" * 70)
    for c in clips:
        print(f"[{c.clip_id}/{len(clips)}] \"{c.title}\" | Score: {c.viral_score}/100 | {c.duration:.1f}s")
        print(f"     Time: {format_seconds_to_timestamp(c.start_time)} -> {format_seconds_to_timestamp(c.end_time)}")
        print(f"     Hook: \"{c.hook}\"")
        print(f"     Why : {c.reason}\n")

    # 2. Process each clip through Reframe + Caption Pipeline (pipelined)
    generated_reels, metadata_list = process_clips_pipelined(
        clips=clips,
        src_video=input_path,
        output_dir=output_dir,
        whisper_clip_model=args.whisper_clip_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        smooth_alpha=args.smooth,
        chunk_size=args.chunk_size,
        bottom_offset=args.bottom_offset,
        save_intermediate=args.save_intermediate,
        no_llm=args.no_llm,
        style=args.style,
        reframe_preset=args.reframe_preset,
        layout=args.layout,
        audio_delay_ms=args.audio_delay,
        aspect=args.aspect,
        no_caption=args.no_caption,
        refine_boundaries=getattr(args, "refine_boundaries", False),
        profile=getattr(args, "profile", False),
        whisper_backend=getattr(args, "whisper_backend", "auto"),
        legacy_render=getattr(args, "legacy_render", False),
        fast=getattr(args, "fast", False),
        detect_every=getattr(args, "detect_every", 3),
        punch_in=getattr(args, "punch_in", "off"),
        detector=getattr(args, "detector", "haar"),
        model_path=getattr(args, "model_path", None),
    )

    # 3. Export Social Media Metadata JSON
    metadata_path = output_dir / "reels_metadata.json"
    metadata_path.write_text(json.dumps(metadata_list, indent=2, ensure_ascii=False), encoding="utf-8")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETE!")
    print("=" * 70)
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Generated {len(generated_reels)} ready-to-post Reels in: {output_dir}")
    for idx, r in enumerate(generated_reels, 1):
        print(f"  {idx}. {r.name}")
    print(f"\nMetadata and post captions saved to: {metadata_path.name}")
    print("=" * 70)


def handle_detect(args: argparse.Namespace) -> None:
    """Only detect and display viral hooks without rendering video."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"[ERROR] Input video not found: {input_path}")

    finder = HookFinder(
        whisper_model=args.whisper_full_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        backend=getattr(args, "whisper_backend", "auto"),
        filter_face_presence=getattr(args, "filter_face_presence", False),
    )
    clips = finder.find_viral_clips(input_path, num_clips=args.clips)

    print("\n" + "=" * 70)
    print(f" AI DETECTED {len(clips)} VIRAL HOOKS IN: {input_path.name}")
    print("=" * 70)
    for c in clips:
        print(f"• Clip #{c.clip_id}: \"{c.title}\" (Viral Score: {c.viral_score}/100)")
        print(f"  Timestamp : {format_seconds_to_timestamp(c.start_time)} -> {format_seconds_to_timestamp(c.end_time)} ({c.duration:.1f}s)")
        print(f"  Hook Line : \"{c.hook}\"")
        print(f"  Rationale : {c.reason}\n")

    if args.json_out:
        out_p = Path(args.json_out).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps([c.model_dump() for c in clips], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Exported JSON metadata to: {out_p}")

    if getattr(args, "interactive", False):
        handle_interactive(args, existing_clips=clips)


def handle_interactive(args: argparse.Namespace, existing_clips: Optional[List[ViralClipMetadata]] = None) -> None:
    """Interactive clip and aspect ratio selection with boundary re-verification."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"[ERROR] Input video not found: {input_path}")

    output_dir = Path(getattr(args, "output_dir", "output")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if existing_clips:
        clips = existing_clips
    else:
        finder = HookFinder(
            whisper_model=getattr(args, "whisper_full_model", "base"),
            llm_model=getattr(args, "llm_model", None),
            api_key=getattr(args, "api_key", None),
            min_duration=getattr(args, "min_duration", 28.0),
            max_duration=getattr(args, "max_duration", 60.0),
            backend=getattr(args, "whisper_backend", "auto"),
            filter_face_presence=getattr(args, "filter_face_presence", False),
        )
        clips = finder.find_viral_clips(input_path, num_clips=getattr(args, "clips", 3))

    if not clips:
        sys.exit("[ERROR] No highlight hooks detected.")

    print("\n" + "=" * 70)
    print(f" INTERACTIVE CLIP & ASPECT RATIO STUDIO: {input_path.name}")
    print("=" * 70)
    for c in clips:
        print(f"[{c.clip_id}] \"{c.title}\" | Score: {c.viral_score}/100 | {c.duration:.1f}s")
        print(f"    Time: {format_seconds_to_timestamp(c.start_time)} -> {format_seconds_to_timestamp(c.end_time)}")
        print(f"    Hook: \"{c.hook}\"")
        print(f"    Why : {c.reason}\n")
    print("=" * 70)

    # 1. Select clip to render
    while True:
        try:
            raw_sel = input("Select clip to render (or 'all', 'q' to quit) [default: 1]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return

        if raw_sel in ("q", "quit", "exit"):
            print("Exiting interactive mode.")
            return
        if not raw_sel or raw_sel == "1":
            selected_clips = [clips[0]]
            break
        elif raw_sel == "all":
            selected_clips = clips
            break
        else:
            try:
                chosen_ids = [int(x.strip()) for x in raw_sel.split(",") if x.strip()]
                selected_clips = [c for c in clips if c.clip_id in chosen_ids]
                if selected_clips:
                    break
                print(f"Invalid ID(s). Please choose between 1 and {len(clips)}.")
            except ValueError:
                print("Please enter a valid clip number (e.g. 1), comma-separated numbers (e.g. 1,2), or 'all'.")

    # 2. Select aspect ratio preset
    print("\nSelect aspect ratio preset:")
    print("  1) 9:16 (Full Vertical Reel - Face Centered)")
    print("  2) 1:1 with Blur B&W Canvas (Aesthetic 9:16 background, square 1:1 video centered)")
    print("  3) 4:5 (Instagram Portrait Feed)")
    print("  4) 16:9 (Native Horizontal - No crop)")

    while True:
        try:
            raw_ar = input("Enter choice [1-4, default: 1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return

        if not raw_ar or raw_ar == "1":
            chosen_aspect = "9:16"
            chosen_layout = "direct-crop"
            break
        elif raw_ar == "2":
            chosen_aspect = "9:16"
            chosen_layout = "blur-bw"
            break
        elif raw_ar == "3":
            chosen_aspect = "4:5"
            chosen_layout = "direct-crop"
            break
        elif raw_ar == "4":
            chosen_aspect = "16:9"
            chosen_layout = "direct-crop"
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    print(f"\nTarget setting: Aspect={chosen_aspect} | Layout={chosen_layout}")
    print(f"Beginning high-precision boundary re-verification and rendering for {len(selected_clips)} clip(s)...")

    rendered_files, _ = process_clips_pipelined(
        clips=selected_clips,
        src_video=input_path,
        output_dir=output_dir,
        whisper_clip_model=getattr(args, "whisper_clip_model", "large-v3"),
        llm_model=getattr(args, "llm_model", None),
        api_key=getattr(args, "api_key", None),
        smooth_alpha=getattr(args, "smooth", 0.06),
        chunk_size=getattr(args, "chunk_size", 4),
        bottom_offset=getattr(args, "bottom_offset", 380),
        save_intermediate=getattr(args, "save_intermediate", False),
        no_llm=getattr(args, "no_llm", False),
        style=getattr(args, "style", "classic-yellow"),
        reframe_preset=getattr(args, "reframe_preset", "talking_head"),
        layout=chosen_layout,
        audio_delay_ms=getattr(args, "audio_delay", 500.0),
        aspect=chosen_aspect,
        no_caption=getattr(args, "no_caption", False),
        refine_boundaries=True,
        profile=getattr(args, "profile", False),
        whisper_backend=getattr(args, "whisper_backend", "auto"),
        legacy_render=getattr(args, "legacy_render", False),
        fast=getattr(args, "fast", False),
        detect_every=getattr(args, "detect_every", 3),
        punch_in=getattr(args, "punch_in", "off"),
        detector=getattr(args, "detector", "haar"),
        model_path=getattr(args, "model_path", None),
    )

    print("\n" + "=" * 70)
    print(" INTERACTIVE RENDERING COMPLETE!")
    print("=" * 70)
    for idx, f in enumerate(rendered_files, 1):
        print(f"  {idx}. {f.name} ({f.stat().st_size / (1024*1024):.1f} MB) -> {f}")
    print("=" * 70)


def handle_clip(args: argparse.Namespace) -> None:
    """Process a single manual timestamp range."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        sys.exit(f"[ERROR] Input video not found: {input_path}")

    start_t = float(args.start)
    end_t = float(args.end)
    if end_t <= start_t:
        sys.exit(f"[ERROR] End time ({end_t}) must be greater than start time ({start_t}).")

    dur = round(end_t - start_t, 2)
    target_aspect = args.aspect
    if target_aspect == "auto":
        target_aspect = "16:9" if dur >= 180.0 else "9:16"

    if args.out:
        out_file = Path(args.out).expanduser().resolve()
    else:
        prefix = "clip_16x9" if target_aspect == "16:9" else "reel"
        out_file = input_path.with_name(f"{prefix}_{input_path.stem}_custom.mp4")
    out_dir = out_file.parent

    clip_meta = ViralClipMetadata(
        clip_id=1,
        title=args.title or ("Custom 16:9 Highlight" if target_aspect == "16:9" else "Custom Highlight Reel"),
        hook=args.hook or "Key Insight",
        start_time=start_t,
        end_time=end_t,
        duration=dur,
        viral_score=90,
        reason="Manual selection",
        slug=out_file.stem,
        aspect_ratio=target_aspect
    )

    process_single_clip(
        src_video=input_path,
        clip_info=clip_meta,
        output_dir=out_dir,
        output_path=out_file,
        whisper_clip_model=args.whisper_clip_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        smooth_alpha=args.smooth,
        chunk_size=args.chunk_size,
        bottom_offset=args.bottom_offset,
        save_intermediate=args.save_intermediate,
        no_llm=args.no_llm,
        style=args.style,
        reframe_preset=args.reframe_preset,
        layout=args.layout,
        audio_delay_ms=args.audio_delay,
        aspect=args.aspect,
        no_caption=args.no_caption,
        refine_boundaries=getattr(args, "refine_boundaries", False),
        profile=getattr(args, "profile", False),
        whisper_backend=getattr(args, "whisper_backend", "auto"),
        legacy_render=getattr(args, "legacy_render", False),
        fast=getattr(args, "fast", False),
        detect_every=getattr(args, "detect_every", 3),
        punch_in=getattr(args, "punch_in", "off"),
        detector=getattr(args, "detector", "haar"),
        model_path=getattr(args, "model_path", None),
    )


def handle_reframe(args: argparse.Namespace) -> None:
    """Execute standalone reframing only."""
    src_path = Path(args.input).expanduser().resolve()
    if not src_path.is_file():
        sys.exit(f"[ERROR] Input video not found: {src_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else src_path.with_name(f"{src_path.stem}_reframe.mp4")

    print("=" * 65)
    print(f" EXECUTING REFRAME WORKFLOW: {src_path.name}")
    print("=" * 65)

    reframe_cfg = ReframeConfig(
        src=src_path,
        out=out_path,
        cascade_path=Path(args.cascade) if args.cascade else None,
        csv_path=Path(args.csv) if args.csv else None,
        smooth_alpha=args.smooth,
        motion_thr=args.motion_thr,
        static_std=args.static_std,
        roi_xmax=args.roi_xmax,
        debug=args.debug,
        preset=args.preset,
        deadzone_px=args.deadzone,
        zoom=args.zoom,
        headroom=args.headroom,
        layout=args.layout,
        audio_delay_ms=args.audio_delay,
        detect_every=getattr(args, "detect_every", 3),
        detect_width=getattr(args, "detect_width", 320),
        punch_in=getattr(args, "punch_in", "off"),
        detector=getattr(args, "detector", "haar"),
        model_path=getattr(args, "model_path", None),
    )
    reframer = FaceReframer(reframe_cfg)
    reframer.run(fast=getattr(args, "fast", False))


def handle_caption(args: argparse.Namespace) -> None:
    """Execute standalone word captioning on existing vertical video."""
    src_path = Path(args.input).expanduser().resolve()
    if not src_path.is_file():
        sys.exit(f"[ERROR] Input video not found: {src_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else src_path.with_name(f"{src_path.stem}_captioned.mp4")

    print("=" * 65)
    print(f" EXECUTING CAPTION WORKFLOW: {src_path.name}")
    print("=" * 65)

    script_hint = (
        Path(args.script).expanduser().read_text(encoding="utf-8").strip()
        if args.script and Path(args.script).is_file()
        else ""
    )
    transcriber = HinglishTranscriber(
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        rpm=args.rpm,
        no_llm=args.no_llm,
        uppercase=not args.no_upper,
        backend=getattr(args, "whisper_backend", "auto"),
    )
    words = transcriber.transcribe_and_romanize(
        video_path=src_path,
        glossary=args.glossary or "",
        script_hint=script_hint,
        whisper_prompt=args.whisper_prompt
    )

    if args.preview:
        print(f"\n[Preview: Formatted Word Chunks | Style: {args.style}]")
        renderer = CaptionRenderer(chunk_size=args.chunk_size, style=args.style)
        chunks = renderer.build_chunks(words)
        for c in chunks[:25]:
            chunk_text = " ".join(w.word for w in c.words)
            print(f"  [{c.start:05.2f}s - {c.end:05.2f}s] {chunk_text}")
        print("\n--preview specified: skipping video render.")
        return

    renderer = CaptionRenderer(
        font_path=args.font,
        chunk_size=args.chunk_size,
        bottom_offset=args.bottom_offset,
        style=args.style
    )
    renderer.render_to_video(
        input_video=src_path,
        output_video=out_path,
        words=words
    )


def build_pipeline_parser() -> argparse.ArgumentParser:
    """Build unified CLI parser for pipeline.py."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Unified High-Performance Lecture to 9:16 Viral Reels Pipeline for Apple Silicon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline.py interactive --input lecture.mp4        # Interactive clip & aspect ratio studio
  python3 pipeline.py run --input lecture.mp4 --clips 3      # Automated end-to-end pipeline
  python3 pipeline.py detect --input lecture.mp4 -I          # Detect hooks + interactive render
  python3 pipeline.py clip --input lecture.mp4 --start 15.0 --end 48.0 --refine-boundaries
  python3 pipeline.py reframe --input lecture.mp4 -o vertical.mp4
  python3 pipeline.py caption --input vertical.mp4 -o captioned.mp4
  python3 pipeline.py test
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline action")

    # Command: 'run'
    p_run = subparsers.add_parser("run", help="Run full automated pipeline (detect + cut + reframe + caption)")
    p_run.add_argument("--input", "-i", required=True, help="Path to input lecture video")
    p_run.add_argument("--clips", "-c", type=int, default=3, help="Number of viral clips to generate (default: 3)")
    p_run.add_argument("--output-dir", "-o", default="output", help="Output directory for generated reels (default: output/)")
    p_run.add_argument("--whisper-full-model", default="base", help="Whisper model for full lecture scan (default: base)")
    p_run.add_argument("--whisper-clip-model", default="large-v3", help="Whisper model for word sync on clip (default: large-v3)")
    p_run.add_argument("--llm-model", default=os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite"), help="Gemini model")
    p_run.add_argument("--api-key", default=None, help="Gemini API key (overrides GEMINI_API_KEY env)")
    p_run.add_argument("--min-duration", type=float, default=28.0, help="Min clip duration in seconds (default: 28.0)")
    p_run.add_argument("--max-duration", type=float, default=60.0, help="Max clip duration in seconds (default: 60.0)")
    p_run.add_argument("--smooth", type=float, default=0.06, help="Face reframing EMA alpha (default: 0.06)")
    p_run.add_argument("--chunk-size", type=int, default=4, help="Caption words per chunk (default: 4)")
    p_run.add_argument("--bottom-offset", type=int, default=380, help="Caption bottom offset in pixels (default: 380)")
    p_run.add_argument("--save-intermediate", action="store_true", help="Keep cut snippets and reframed videos in output/intermediates/")
    p_run.add_argument("--no-llm", action="store_true", help="Use local rule-based romanization only")
    p_run.add_argument("--style", default="classic-yellow", help="Caption visual preset style (default: classic-yellow). Options: classic-yellow, ae-glow, box-highlight, bold-stroke, fade-in, kalakar-fire, karaoke-neon, cyber-neon, cinematic, comic-pop, minimal, subtitle-16x9")
    p_run.add_argument("--reframe-preset", default="talking_head", help="Reframing preset (default: talking_head). Options: talking_head, lecture, smooth, action")
    p_run.add_argument("--layout", default="direct-crop", choices=["direct-crop", "blur-bw"], help="Output canvas layout (default: direct-crop). Options: direct-crop, blur-bw")
    p_run.add_argument("--audio-delay", type=float, default=500.0, help="Audio delay offset in milliseconds to fix lip sync (default: 500.0)")
    p_run.add_argument("--aspect", default="auto", choices=["auto", "9:16", "16:9", "4:5", "9x16", "16x9", "4x5"], help="Target aspect ratio branching: auto (<180s -> 9:16, >=180s -> 16:9), 9:16, 4:5, or 16:9 (default: auto)")
    p_run.add_argument("--no-caption", action="store_true", help="Bypass caption burning and export clean trimmed/reframed video")
    p_run.add_argument("--caption", dest="no_caption", action="store_false", help="Force caption burning (default: enabled)")
    p_run.add_argument("--refine-boundaries", action="store_true", help="Run high-precision sentence boundary re-verification before rendering")
    p_run.add_argument("--profile", action="store_true", help="Print per-stage wall-clock timing profile and save to output/profile_<clip>.json")
    p_run.add_argument("--whisper-backend", default="auto", choices=["auto", "openai", "mlx"], help="Whisper backend engine (default: auto, mlx if installed on Apple Silicon, else openai)")
    p_run.add_argument("--legacy-render", action="store_true", help="Use legacy multi-step encoding (reframe encode -> caption encode) instead of single-pass render_reel")
    p_run.add_argument("--fast", action="store_true", help="Use Apple Silicon hardware acceleration (h264_videotoolbox -b:v 12M) for final encode")
    p_run.add_argument("--detect-every", type=int, default=3, help="Run face detection every Nth frame with template tracking in between (default: 3)")
    p_run.add_argument("--filter-face-presence", action="store_true", help="Filter vertical clips where teacher face is visible in < 70%% of frames")
    p_run.add_argument("--punch-in", default="off", choices=["off", "auto"], help="Dynamic punch-in zoom on emphasis moments & RMS energy peaks (default: off)")
    p_run.add_argument("--detector", default="haar", choices=["haar", "yunet", "vision"], help="Face detector backend: haar (default), yunet (ONNX neural net), or vision (Apple Vision framework)")
    p_run.add_argument("--model-path", default=None, help="Custom path to YuNet ONNX model (face_detection_yunet_2023mar.onnx)")

    # Command: 'interactive'
    p_inter = subparsers.add_parser("interactive", help="Interactive clip and aspect ratio selection with boundary re-verification")
    p_inter.add_argument("--input", "-i", required=True, help="Path to input lecture video")
    p_inter.add_argument("--clips", "-c", type=int, default=5, help="Number of viral hooks to detect (default: 5)")
    p_inter.add_argument("--output-dir", "-o", default="output", help="Output directory for generated clips (default: output/)")
    p_inter.add_argument("--whisper-full-model", default="base", help="Whisper model for full lecture scan (default: base)")
    p_inter.add_argument("--whisper-clip-model", default="large-v3", help="Whisper model for word sync on clip (default: large-v3)")
    p_inter.add_argument("--llm-model", default=os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite"), help="Gemini model")
    p_inter.add_argument("--api-key", default=None, help="Gemini API key")
    p_inter.add_argument("--min-duration", type=float, default=28.0, help="Min clip duration in seconds (default: 28.0)")
    p_inter.add_argument("--max-duration", type=float, default=60.0, help="Max clip duration in seconds (default: 60.0)")
    p_inter.add_argument("--smooth", type=float, default=0.06, help="Face reframing EMA alpha (default: 0.06)")
    p_inter.add_argument("--chunk-size", type=int, default=4, help="Caption words per chunk (default: 4)")
    p_inter.add_argument("--bottom-offset", type=int, default=380, help="Caption bottom offset in pixels (default: 380)")
    p_inter.add_argument("--save-intermediate", action="store_true", help="Keep cut snippets and reframed videos")
    p_inter.add_argument("--no-llm", action="store_true", help="Use local rule-based romanization only")
    p_inter.add_argument("--style", default="classic-yellow", help="Caption visual preset style (default: classic-yellow)")
    p_inter.add_argument("--reframe-preset", default="talking_head", help="Reframing preset (default: talking_head)")
    p_inter.add_argument("--audio-delay", type=float, default=500.0, help="Audio delay offset in milliseconds to fix lip sync (default: 500.0)")
    p_inter.add_argument("--no-caption", action="store_true", help="Bypass caption burning and export clean trimmed/reframed video")
    p_inter.add_argument("--profile", action="store_true", help="Print per-stage wall-clock timing profile and save to output/profile_<clip>.json")
    p_inter.add_argument("--whisper-backend", default="auto", choices=["auto", "openai", "mlx"], help="Whisper backend engine (default: auto, mlx if installed on Apple Silicon, else openai)")
    p_inter.add_argument("--legacy-render", action="store_true", help="Use legacy multi-step encoding (reframe encode -> caption encode) instead of single-pass render_reel")
    p_inter.add_argument("--fast", action="store_true", help="Use Apple Silicon hardware acceleration (h264_videotoolbox -b:v 12M) for final encode")
    p_inter.add_argument("--detect-every", type=int, default=3, help="Run face detection every Nth frame with template tracking in between (default: 3)")
    p_inter.add_argument("--filter-face-presence", action="store_true", help="Filter vertical clips where teacher face is visible in < 70%% of frames")
    p_inter.add_argument("--punch-in", default="off", choices=["off", "auto"], help="Dynamic punch-in zoom on emphasis moments & RMS energy peaks (default: off)")
    p_inter.add_argument("--detector", default="haar", choices=["haar", "yunet", "vision"], help="Face detector backend: haar (default), yunet (ONNX neural net), or vision (Apple Vision framework)")
    p_inter.add_argument("--model-path", default=None, help="Custom path to YuNet ONNX model (face_detection_yunet_2023mar.onnx)")

    # Command: 'detect'
    p_detect = subparsers.add_parser("detect", help="Detect viral moments & hooks without rendering video")
    p_detect.add_argument("--input", "-i", required=True, help="Path to input lecture video")
    p_detect.add_argument("--clips", "-c", type=int, default=3, help="Number of hooks to detect (default: 3)")
    p_detect.add_argument("--whisper-full-model", default="base", help="Whisper model for full lecture scan (default: base)")
    p_detect.add_argument("--llm-model", default=os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite"), help="Gemini model")
    p_detect.add_argument("--api-key", default=None, help="Gemini API key")
    p_detect.add_argument("--min-duration", type=float, default=28.0, help="Min duration (default: 28.0)")
    p_detect.add_argument("--max-duration", type=float, default=60.0, help="Max duration (default: 60.0)")
    p_detect.add_argument("--json-out", default=None, help="Optional path to export detected hooks as JSON")
    p_detect.add_argument("--interactive", "-I", action="store_true", help="Launch interactive clip & aspect ratio selector after detection")
    p_detect.add_argument("--output-dir", "-o", default="output", help="Output directory for generated clips (default: output/)")
    p_detect.add_argument("--whisper-clip-model", default="large-v3", help="Whisper model for word sync (default: large-v3)")
    p_detect.add_argument("--style", default="classic-yellow", help="Caption visual preset style (default: classic-yellow)")
    p_detect.add_argument("--reframe-preset", default="talking_head", help="Reframing preset (default: talking_head)")
    p_detect.add_argument("--smooth", type=float, default=0.06, help="Face reframing EMA alpha (default: 0.06)")
    p_detect.add_argument("--chunk-size", type=int, default=4, help="Caption words per chunk (default: 4)")
    p_detect.add_argument("--bottom-offset", type=int, default=380, help="Caption bottom offset in pixels (default: 380)")
    p_detect.add_argument("--audio-delay", type=float, default=500.0, help="Audio delay offset in milliseconds (default: 500.0)")
    p_detect.add_argument("--save-intermediate", action="store_true", help="Keep cut snippets and reframed videos")
    p_detect.add_argument("--no-llm", action="store_true", help="Use local rule-based romanization only")
    p_detect.add_argument("--no-caption", action="store_true", help="Bypass caption burning")
    p_detect.add_argument("--whisper-backend", default="auto", choices=["auto", "openai", "mlx"], help="Whisper backend engine (default: auto, mlx if installed on Apple Silicon, else openai)")
    p_detect.add_argument("--filter-face-presence", action="store_true", help="Filter vertical clips where teacher face is visible in < 70%% of frames")
    p_detect.add_argument("--detector", default="haar", choices=["haar", "yunet", "vision"], help="Face detector backend: haar (default), yunet (ONNX neural net), or vision (Apple Vision framework)")
    p_detect.add_argument("--model-path", default=None, help="Custom path to YuNet ONNX model (face_detection_yunet_2023mar.onnx)")

    # Command: 'clip'
    p_clip = subparsers.add_parser("clip", help="Render a specific manual time window end-to-end")
    p_clip.add_argument("--input", "-i", required=True, help="Path to input lecture video")
    p_clip.add_argument("--start", type=float, required=True, help="Start time in seconds")
    p_clip.add_argument("--end", type=float, required=True, help="End time in seconds")
    p_clip.add_argument("-o", "--out", default=None, help="Path to output reel MP4")
    p_clip.add_argument("--title", default=None, help="Title for the reel")
    p_clip.add_argument("--hook", default=None, help="Hook description")
    p_clip.add_argument("--whisper-clip-model", default="large-v3", help="Whisper model for word timestamps (default: large-v3)")
    p_clip.add_argument("--llm-model", default=os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite"), help="Gemini model")
    p_clip.add_argument("--api-key", default=None, help="Gemini API key")
    p_clip.add_argument("--smooth", type=float, default=0.06, help="EMA alpha factor (default: 0.06)")
    p_clip.add_argument("--chunk-size", type=int, default=4, help="Caption chunk size (default: 4)")
    p_clip.add_argument("--bottom-offset", type=int, default=380, help="Caption bottom offset (default: 380)")
    p_clip.add_argument("--save-intermediate", action="store_true", help="Save intermediate snippet and reframed video")
    p_clip.add_argument("--no-llm", action="store_true", help="Use local rule-based romanizer only")
    p_clip.add_argument("--style", default="classic-yellow", help="Caption visual preset style (default: classic-yellow). Options: classic-yellow, ae-glow, box-highlight, bold-stroke, fade-in, kalakar-fire, karaoke-neon, cyber-neon, cinematic, comic-pop, minimal, subtitle-16x9")
    p_clip.add_argument("--reframe-preset", default="talking_head", help="Reframing preset (default: talking_head)")
    p_clip.add_argument("--layout", default="direct-crop", choices=["direct-crop", "blur-bw"], help="Output canvas layout (default: direct-crop). Options: direct-crop, blur-bw")
    p_clip.add_argument("--audio-delay", type=float, default=500.0, help="Audio delay offset in milliseconds to fix lip sync (default: 500.0)")
    p_clip.add_argument("--aspect", default="auto", choices=["auto", "9:16", "16:9", "4:5", "9x16", "16x9", "4x5"], help="Target aspect ratio branching: auto (<180s -> 9:16, >=180s -> 16:9), 9:16, 4:5, or 16:9 (default: auto)")
    p_clip.add_argument("--no-caption", action="store_true", help="Bypass caption burning and export clean trimmed/reframed video")
    p_clip.add_argument("--caption", dest="no_caption", action="store_false", help="Force caption burning (default: enabled)")
    p_clip.add_argument("--refine-boundaries", action="store_true", help="Run high-precision sentence boundary re-verification before rendering")
    p_clip.add_argument("--profile", action="store_true", help="Print per-stage wall-clock timing profile and save to output/profile_<clip>.json")
    p_clip.add_argument("--whisper-backend", default="auto", choices=["auto", "openai", "mlx"], help="Whisper backend engine (default: auto, mlx if installed on Apple Silicon, else openai)")
    p_clip.add_argument("--legacy-render", action="store_true", help="Use legacy multi-step encoding (reframe encode -> caption encode) instead of single-pass render_reel")
    p_clip.add_argument("--fast", action="store_true", help="Use Apple Silicon hardware acceleration (h264_videotoolbox -b:v 12M) for final encode")
    p_clip.add_argument("--detect-every", type=int, default=3, help="Run face detection every Nth frame with template tracking in between (default: 3)")
    p_clip.add_argument("--punch-in", default="off", choices=["off", "auto"], help="Dynamic punch-in zoom on emphasis moments & RMS energy peaks (default: off)")
    p_clip.add_argument("--detector", default="haar", choices=["haar", "yunet", "vision"], help="Face detector backend: haar (default), yunet (ONNX neural net), or vision (Apple Vision framework)")
    p_clip.add_argument("--model-path", default=None, help="Custom path to YuNet ONNX model (face_detection_yunet_2023mar.onnx)")

    # Command: 'reframe'
    p_reframe = subparsers.add_parser("reframe", help="Standalone face-tracking 9:16 vertical crop")
    p_reframe.add_argument("--input", "-i", required=True, help="Path to input horizontal video")
    p_reframe.add_argument("-o", "--out", default=None, help="Path to output vertical video")
    p_reframe.add_argument("--preset", default="talking_head", help="Reframing preset (default: talking_head). Options: talking_head, lecture, smooth, action")
    p_reframe.add_argument("--layout", default="direct-crop", choices=["direct-crop", "blur-bw"], help="Output canvas layout (default: direct-crop). Options: direct-crop, blur-bw")
    p_reframe.add_argument("--audio-delay", type=float, default=0.0, help="Audio delay offset in milliseconds (default: 0.0)")
    p_reframe.add_argument("--cascade", default=None, help="Path to haarcascade_frontalface_default.xml")
    p_reframe.add_argument("--csv", default=None, help="Path to export crop keyframes CSV")
    p_reframe.add_argument("--smooth", type=float, default=None, help="EMA alpha factor (default: from preset)")
    p_reframe.add_argument("--deadzone", type=float, default=None, help="Dead-zone dampening in pixels (default: from preset)")
    p_reframe.add_argument("--zoom", type=float, default=None, help="Framing zoom factor (e.g. 1.10)")
    p_reframe.add_argument("--headroom", type=float, default=None, help="Headroom ratio from top (default: 0.18)")
    p_reframe.add_argument("--motion-thr", type=float, default=0.03, help="Motion threshold (default: 0.03)")
    p_reframe.add_argument("--static-std", type=float, default=8.0, help="Static std threshold (default: 8.0)")
    p_reframe.add_argument("--roi-xmax", type=float, default=1.0, help="Ignore detections to the right (default: 1.0)")
    p_reframe.add_argument("--debug", action="store_true", help="Generate debug video overlay")
    p_reframe.add_argument("--fast", action="store_true", help="Use Apple Silicon hardware acceleration (h264_videotoolbox -b:v 12M) for final encode")
    p_reframe.add_argument("--detect-every", type=int, default=3, help="Run face detection every Nth frame with template tracking in between (default: 3)")
    p_reframe.add_argument("--detect-width", type=int, default=320, help="Downscale width for face detection (default: 320)")
    p_reframe.add_argument("--punch-in", default="off", choices=["off", "auto"], help="Dynamic punch-in zoom on emphasis moments & RMS energy peaks (default: off)")
    p_reframe.add_argument("--detector", default="haar", choices=["haar", "yunet", "vision"], help="Face detector backend: haar (default), yunet (ONNX neural net), or vision (Apple Vision framework)")
    p_reframe.add_argument("--model-path", default=None, help="Custom path to YuNet ONNX model (face_detection_yunet_2023mar.onnx)")

    # Command: 'caption'
    p_caption = subparsers.add_parser("caption", help="Standalone word-by-word captions on vertical video")
    p_caption.add_argument("--input", "-i", required=True, help="Path to input vertical video")
    p_caption.add_argument("-o", "--out", default=None, help="Path to output captioned video")
    p_caption.add_argument("--whisper-model", default="large-v3", help="Whisper model (default: large-v3)")
    p_caption.add_argument("--llm-model", default=os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite"), help="Gemini model")
    p_caption.add_argument("--api-key", default=None, help="Gemini API key")
    p_caption.add_argument("--rpm", type=float, default=8.0, help="Max requests per minute (default: 8.0)")
    p_caption.add_argument("--script", default=None, help="Path to optional reference script text file")
    p_caption.add_argument("--glossary", default="", help="Optional comma-separated domain terms")
    p_caption.add_argument("--whisper-prompt", default=None, help="Optional Whisper initial prompt")
    p_caption.add_argument("--chunk-size", type=int, default=4, help="Words per caption chunk (default: 4)")
    p_caption.add_argument("--bottom-offset", type=int, default=380, help="Caption bottom offset in px (default: 380)")
    p_caption.add_argument("--font", default=None, help="Path to custom TrueType font")
    p_caption.add_argument("--style", default="classic-yellow", help="Caption visual preset style (default: classic-yellow). Options: classic-yellow, ae-glow, box-highlight, bold-stroke, fade-in, kalakar-fire, karaoke-neon, cyber-neon, cinematic, comic-pop, minimal, subtitle-16x9")
    p_caption.add_argument("--no-llm", action="store_true", help="Use local rule-based romanizer only")
    p_caption.add_argument("--no-upper", action="store_true", help="Keep text lowercase")
    p_caption.add_argument("--preview", action="store_true", help="Transcribe & print preview only without rendering")
    p_caption.add_argument("--whisper-backend", default="auto", choices=["auto", "openai", "mlx"], help="Whisper backend engine (default: auto, mlx if installed on Apple Silicon, else openai)")
    p_caption.add_argument("--fast", action="store_true", help="Use Apple Silicon hardware acceleration (h264_videotoolbox -b:v 12M) for final encode")

    # Command: 'test'
    p_test = subparsers.add_parser("test", help="Run system diagnostics and health check")
    p_test.add_argument("--llm-model", default=os.getenv("HINGLISH_MODEL", "gemini-3.5-flash-lite"), help="Model to verify")

    return parser


def main() -> None:
    parser = build_pipeline_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.command == "interactive":
        handle_interactive(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "detect":
        handle_detect(args)
    elif args.command == "clip":
        handle_clip(args)
    elif args.command == "reframe":
        handle_reframe(args)
    elif args.command == "caption":
        handle_caption(args)
    elif args.command == "test":
        run_diagnostics(model_name=args.llm_model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
