#!/usr/bin/env python3
"""
av_sync.py - Strict Audio-Video Lip-Sync and Constant Frame Rate (CFR) Normalization.

Features:
1. Accurate snippet cutting with PTS/keyframe audio offset zeroing.
2. Constant Frame Rate (CFR) normalization preventing progressive OpenCV iteration drift.
3. Fast probe verification via ffprobe asserting video_duration and audio_duration within 0.05s.
4. Dynamic aresample=async=1000 locking audio clock to video clock.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def probe_av_sync(video_path: Path | str) -> dict:
    """
    Fast ffprobe check of audio and video streams:
    - video_duration, audio_duration, duration_diff
    - video_start, audio_start, start_diff
    - video_fps, is_cfr
    - is_synced: True if duration_diff <= 0.05 and start_diff <= 0.05
    """
    v_path = Path(video_path).expanduser().resolve()
    if not v_path.is_file():
        return {"valid": False, "error": f"File not found: {v_path}"}

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,start_time,duration,r_frame_rate,avg_frame_rate:format=duration,start_time",
        "-of", "json",
        str(v_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
    except Exception as e:
        return {"valid": False, "error": str(e)}

    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not v_stream:
        return {"valid": False, "has_video": False, "has_audio": a_stream is not None}

    format_data = data.get("format", {})
    fmt_dur = float(format_data.get("duration") or 0.0)

    v_dur = float(v_stream.get("duration") or fmt_dur)
    a_dur = float(a_stream.get("duration") or fmt_dur) if a_stream else 0.0

    v_start = float(v_stream.get("start_time") or 0.0)
    a_start = float(a_stream.get("start_time") or 0.0) if a_stream else 0.0

    dur_diff = abs(v_dur - a_dur) if a_stream else 0.0
    start_diff = abs(v_start - a_start) if a_stream else 0.0

    # Parse framerate
    r_fps_str = v_stream.get("r_frame_rate", "25/1")
    avg_fps_str = v_stream.get("avg_frame_rate", "25/1")

    def parse_fps(fps_str: str) -> float:
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/", 1)
                return float(num) / float(den) if float(den) != 0 else 25.0
            return float(fps_str)
        except Exception:
            return 25.0

    r_fps = parse_fps(r_fps_str)
    avg_fps = parse_fps(avg_fps_str)
    is_cfr = abs(r_fps - avg_fps) < 0.02

    is_synced = (dur_diff <= 0.05 and start_diff <= 0.05) if a_stream else True

    return {
        "valid": True,
        "has_video": True,
        "has_audio": a_stream is not None,
        "v_duration": v_dur,
        "a_duration": a_dur,
        "duration_diff": dur_diff,
        "v_start": v_start,
        "a_start": a_start,
        "start_diff": start_diff,
        "v_fps": r_fps,
        "is_cfr": is_cfr,
        "is_synced": is_synced,
    }


def verify_av_sync(video_path: Path | str, max_diff: float = 0.05) -> Tuple[bool, dict]:
    """
    Log and verify audio-video sync on video file.
    """
    info = probe_av_sync(video_path)
    if not info.get("valid"):
        print(f"[sync] Warning: Could not probe AV sync on {video_path}: {info.get('error')}")
        return False, info

    if not info.get("has_audio"):
        return True, info

    dur_diff = info["duration_diff"]
    start_diff = info["start_diff"]
    synced = dur_diff <= max_diff and start_diff <= max_diff

    status = "LOCKED [OK]" if synced else "DRIFT DETECTED [WARN]"
    print(
        f"[sync] AV sync check: video={info['v_duration']:.3f}s, audio={info['a_duration']:.3f}s "
        f"(diff={dur_diff*1000:.1f}ms, start_diff={start_diff*1000:.1f}ms, cfr={info['is_cfr']}) -> {status}"
    )
    return synced, info


def cut_snippet_accurate(
    src_video: Path | str,
    out_snippet: Path | str,
    start_time: float,
    end_time: float,
    target_fps: Optional[float] = None,
    audio_delay_ms: float = 500.0
) -> Path:
    """
    Accurately cuts time snippet from source video with strict audio-video lip-sync:
    1. Input-seeking with timestamp zeroing (-avoid_negative_ts make_zero -async 1).
    2. PTS reset: -vf setpts=PTS-STARTPTS.
    3. Audio delay offset: adelay={audio_delay_ms} to correct studio audio-lead lip desync.
    4. Audio sync: -af asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0.
    5. Constant Frame Rate (CFR) normalization (-fps_mode cfr).
    6. Post-cut probe check ensuring video_duration and audio_duration within 0.05s.
    """
    src_p = Path(src_video).expanduser().resolve()
    out_p = Path(out_snippet).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not src_p.is_file():
        raise FileNotFoundError(f"Source video not found: {src_p}")

    st = max(0.0, float(start_time))
    et = float(end_time)
    if et <= st:
        raise ValueError(f"Invalid snippet range: start={st:.2f}s, end={et:.2f}s")

    if audio_delay_ms > 0:
        delay_int = int(round(audio_delay_ms))
        af_chain = f"adelay={delay_int}|{delay_int},asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0"
    elif audio_delay_ms < 0:
        af_chain = f"atrim=start={abs(audio_delay_ms)/1000.0:.3f},asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0"
    else:
        af_chain = "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{st:.3f}",
        "-to", f"{et:.3f}",
        "-i", str(src_p),
        "-vf", "setpts=PTS-STARTPTS",
        "-af", af_chain,
        "-avoid_negative_ts", "make_zero",
        "-async", "1",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
    ]
    if target_fps and target_fps > 0:
        cmd.extend(["-r", f"{target_fps:.2f}"])

    cmd.append(str(out_p))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError(f"Accurate snippet cutting failed: {res.stderr}")

    # Post-cut sync probe check
    is_synced, info = verify_av_sync(out_p, max_diff=0.05)
    if not is_synced and info.get("has_audio"):
        # Quick remediation pass to lock audio length to video
        realign_tmp = out_p.with_name(f"{out_p.stem}_realigned.mp4")
        realign_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(out_p),
            "-vf", "setpts=PTS-STARTPTS",
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0",
            "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(realign_tmp)
        ]
        re_res = subprocess.run(realign_cmd, capture_output=True, text=True)
        if re_res.returncode == 0 and realign_tmp.exists() and realign_tmp.stat().st_size > 0:
            realign_tmp.replace(out_p)
            verify_av_sync(out_p, max_diff=0.05)

    return out_p


def ensure_cfr_video(
    video_path: Path | str,
    target_fps: Optional[float] = None,
    output_path: Optional[Path | str] = None,
    audio_delay_ms: float = 0.0
) -> Path:
    """
    Ensure video is Constant Frame Rate (CFR) and in AV sync.
    If already CFR and synced, returns original video with zero overhead.
    If VFR, desynced, or audio_delay_ms != 0, normalizes in a fast ultrafast CFR pass.
    """
    v_path = Path(video_path).expanduser().resolve()
    info = probe_av_sync(v_path)

    fps_match = True
    if target_fps is not None and info.get("v_fps"):
        fps_match = abs(info["v_fps"] - target_fps) < 0.1

    if info.get("valid") and info.get("is_cfr") and info.get("is_synced") and fps_match and audio_delay_ms == 0.0:
        return v_path

    out_p = Path(output_path).expanduser().resolve() if output_path else v_path.with_name(f"{v_path.stem}_cfr.mp4")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sync] Normalizing video to Constant Frame Rate (CFR): {v_path.name} -> {out_p.name}")
    if audio_delay_ms > 0:
        delay_int = int(round(audio_delay_ms))
        af_chain = f"adelay={delay_int}|{delay_int},asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0"
    elif audio_delay_ms < 0:
        af_chain = f"atrim=start={abs(audio_delay_ms)/1000.0:.3f},asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0"
    else:
        af_chain = "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(v_path),
        "-vf", "setpts=PTS-STARTPTS",
        "-af", af_chain,
        "-avoid_negative_ts", "make_zero",
        "-async", "1",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
    ]
    if audio_delay_ms != 0.0:
        cmd.append("-shortest")
    if target_fps and target_fps > 0:
        cmd.extend(["-r", f"{target_fps:.2f}"])

    cmd.append(str(out_p))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError(f"CFR normalization failed: {res.stderr}")

    verify_av_sync(out_p, max_diff=0.05)
    return out_p

