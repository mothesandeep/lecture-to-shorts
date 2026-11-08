#!/usr/bin/env python3
"""
render.py - Unified Single-Pass Audio-Video Reel Renderer.

Eliminates redundant intermediate video re-encodes by combining:
1. Video frame decoding (from high-quality CFR snippet).
2. Per-frame dynamic face-tracking crop & scale (from ReframePlan).
3. Per-frame word-by-word caption composition (from CaptionRenderer composer).
4. Direct streaming into a SINGLE FFmpeg encoder subprocess.

Encode Pipeline Comparison:
- Legacy: snippet encode -> reframer encode -> captioner encode (3 encodes).
- Unified: snippet cut (crf<=12) -> ONE unified render encode (1 final lossy encode).
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

try:
    from core.av_sync import probe_av_sync, verify_av_sync
    from core.captioner import CaptionRenderer, CaptionStyle
    from core.reframer import ReframePlan
    from core.transcriber import WordCaption
except ImportError:
    from av_sync import probe_av_sync, verify_av_sync
    from captioner import CaptionRenderer, CaptionStyle
    from reframer import ReframePlan
    from transcriber import WordCaption


def render_reel(
    snippet: Path | str,
    plan: Optional[ReframePlan] = None,
    words: Optional[List[WordCaption]] = None,
    style: Optional[Union[str, CaptionStyle]] = "classic-yellow",
    out_path: Optional[Path | str] = None,
    chunk_size: int = 4,
    bottom_offset: int = 380,
    font_path: Optional[str | Path] = None,
    crf: Optional[int] = None,
    encoder_preset: Optional[str] = None,
    audio_bitrate: Optional[str] = None,
    no_caption: bool = False,
    fast: bool = False,
) -> Path:
    """
    Render final viral reel in a single pass:
    - Producer thread reads frames from snippet, crops & scales, and composes captions.
    - Decoupled bounded queue (maxsize=8) streams rawvideo bytes.
    - Consumer thread feeds bytes directly to ONE FFmpeg encoder subprocess.
    - Hardware-accelerated Apple Silicon encoding via h264_videotoolbox (-b:v 12M) with --fast.
    - Audio mapped directly from the snippet.
    """
    snippet_p = Path(snippet).expanduser().resolve()
    if not snippet_p.is_file():
        raise FileNotFoundError(f"Source snippet not found: {snippet_p}")

    if out_path:
        out_p = Path(out_path).expanduser().resolve()
    elif plan and plan.out_path:
        out_p = Path(plan.out_path).expanduser().resolve()
    else:
        out_p = snippet_p.with_name(f"{snippet_p.stem}_reel.mp4")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve canvas geometry and reframing parameters
    if plan is not None:
        canvas_w = plan.out_canvas_w
        canvas_h = plan.out_canvas_h
        fps = plan.fps
        crop_w = plan.crop_w
        crop_h = plan.crop_h
        crop_x0 = plan.crop_x0
        crop_y0 = plan.crop_y0
        layout = plan.layout
        eff_crf = crf if crf is not None else getattr(plan, "crf", 17)
        eff_preset = encoder_preset if encoder_preset is not None else getattr(plan, "encoder_preset", "fast")
        eff_audio_bitrate = audio_bitrate if audio_bitrate is not None else getattr(plan, "audio_bitrate", "192k")
    else:
        # Pass-through / native aspect (e.g. 16:9 widescreen)
        cap_probe = cv2.VideoCapture(str(snippet_p))
        if not cap_probe.isOpened():
            raise FileNotFoundError(f"Cannot probe video: {snippet_p}")
        canvas_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        canvas_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
        cap_probe.release()
        crop_w, crop_h = canvas_w, canvas_h
        crop_x0, crop_y0 = None, None
        layout = "direct-crop"
        eff_crf = crf if crf is not None else 17
        eff_preset = encoder_preset if encoder_preset is not None else "fast"
        eff_audio_bitrate = audio_bitrate if audio_bitrate is not None else "192k"

    # 2. Check audio stream existence in snippet
    info = probe_av_sync(snippet_p)
    has_audio = bool(info.get("has_audio", True))

    # 3. Build caption composer closure if captions requested
    composer = None
    if words and not no_caption:
        renderer = CaptionRenderer(
            font_path=font_path,
            chunk_size=chunk_size,
            bottom_offset=bottom_offset,
            style=style,
            crf=eff_crf,
            preset=eff_preset,
        )
        composer = renderer.create_composer(words, canvas_w, canvas_h)
        print(f"[render] Caption composer active: {len(words)} words, style='{renderer.style.name}'")

    # 4. Spawn single FFmpeg encoder subprocess
    enc_desc = "h264_videotoolbox (-b:v 12M)" if fast else f"libx264 (crf={eff_crf}, preset={eff_preset})"
    print(f"[render] Unified single-pass encode -> {out_p.name} ({canvas_w}x{canvas_h} @ {fps:.2f}fps, {enc_desc})")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{canvas_w}x{canvas_h}", "-r", f"{fps}", "-i", "-",
        "-i", str(snippet_p),
    ]
    if has_audio:
        ffmpeg_cmd.extend([
            "-map", "0:v", "-map", "1:a:0",
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0",
            "-c:a", "aac", "-b:a", str(eff_audio_bitrate),
        ])
    else:
        ffmpeg_cmd.extend(["-map", "0:v"])

    if fast:
        ffmpeg_cmd.extend([
            "-fps_mode", "cfr",
            "-c:v", "h264_videotoolbox", "-b:v", "12M", "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            str(out_p),
        ])
    else:
        ffmpeg_cmd.extend([
            "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", eff_preset, "-crf", str(eff_crf), "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            str(out_p),
        ])

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # 5. Decoder, transform, and captioning producer-consumer pipeline
    cap = cv2.VideoCapture(str(snippet_p))
    if not cap.isOpened():
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.kill()
        proc.wait()
        raise FileNotFoundError(f"Cannot open video for reading: {snippet_p}")

    frame_queue: queue.Queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    producer_err: List[Exception] = []
    consumer_err: List[Exception] = []
    frame_count = 0
    t0 = time.time()

    def producer() -> None:
        nonlocal frame_count
        try:
            f_idx = 0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                t = f_idx
                current_time = f_idx / fps
                f_idx += 1
                frame_count = f_idx

                # Transform / Crop / Scale
                if plan is not None and crop_x0 is not None and crop_y0 is not None:
                    x0 = float(crop_x0[t]) if t < len(crop_x0) else float(crop_x0[-1])
                    y0 = float(crop_y0[t]) if t < len(crop_y0) else float(crop_y0[-1])
                    w_t = float(plan.crop_w_arr[t]) if (plan.crop_w_arr is not None and t < len(plan.crop_w_arr)) else float(crop_w)
                    h_t = float(plan.crop_h_arr[t]) if (plan.crop_h_arr is not None and t < len(plan.crop_h_arr)) else float(crop_h)

                    if layout == "blur-bw":
                        fh, fw = frame.shape[:2]
                        scale_bg = max(canvas_w / fw, canvas_h / fh)
                        bg_w, bg_h = int(round(fw * scale_bg)), int(round(fh * scale_bg))
                        bg_scaled = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
                        bx0 = max(0, (bg_w - canvas_w) // 2)
                        by0 = max(0, (bg_h - canvas_h) // 2)
                        bg_crop = bg_scaled[by0:by0 + canvas_h, bx0:bx0 + canvas_w]
                        bg_gray = cv2.cvtColor(bg_crop, cv2.COLOR_BGR2GRAY)
                        bg_blur = cv2.GaussianBlur(bg_gray, (51, 51), 0)
                        canvas_frame = cv2.cvtColor(bg_blur, cv2.COLOR_GRAY2BGR)

                        sx_fg = 1080.0 / w_t
                        sy_fg = 1080.0 / h_t
                        M_fg = np.float32([
                            [sx_fg, 0, -sx_fg * x0],
                            [0, sy_fg, -sy_fg * y0]
                        ])
                        fg_crop = cv2.warpAffine(frame, M_fg, (1080, 1080), flags=cv2.INTER_CUBIC)
                        canvas_frame[420:1500, 0:1080] = fg_crop

                        cv2.line(canvas_frame, (0, 419), (1080, 419), (90, 90, 90), 2)
                        cv2.line(canvas_frame, (0, 1500), (1080, 1500), (90, 90, 90), 2)
                    else:
                        # Single fused crop + scale warp directly to final canvas size
                        sx = float(canvas_w) / w_t
                        sy = float(canvas_h) / h_t
                        M = np.float32([
                            [sx, 0, -sx * x0],
                            [0, sy, -sy * y0]
                        ])
                        canvas_frame = cv2.warpAffine(frame, M, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC)
                else:
                    if frame.shape[1] != canvas_w or frame.shape[0] != canvas_h:
                        canvas_frame = cv2.resize(frame, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
                    else:
                        canvas_frame = frame

                # Compose Captions
                if composer is not None:
                    canvas_frame = composer(canvas_frame, current_time)

                raw_bytes = canvas_frame.tobytes()

                while not stop_event.is_set():
                    try:
                        frame_queue.put(raw_bytes, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            producer_err.append(exc)
            stop_event.set()
        finally:
            if not producer_err:
                while not stop_event.is_set():
                    try:
                        frame_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def consumer() -> None:
        try:
            while not stop_event.is_set():
                try:
                    item = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    frame_queue.task_done()
                    break
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.write(item)
                except (BrokenPipeError, OSError) as exc:
                    consumer_err.append(exc)
                    stop_event.set()
                    frame_queue.task_done()
                    break
                frame_queue.task_done()
        except Exception as exc:
            consumer_err.append(exc)
            stop_event.set()
        finally:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

    producer_thread = threading.Thread(target=producer, name="RenderProducer")
    consumer_thread = threading.Thread(target=consumer, name="RenderConsumer")
    producer_thread.start()
    consumer_thread.start()

    try:
        producer_thread.join()
        consumer_thread.join()
    except BaseException:
        stop_event.set()
        raise
    finally:
        stop_event.set()
        # Drain any remaining items so threads do not stay blocked in put()
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
                frame_queue.task_done()
            except (queue.Empty, ValueError):
                break
        producer_thread.join(timeout=5.0)
        consumer_thread.join(timeout=5.0)
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if producer_err:
        raise producer_err[0]
    if consumer_err and proc.returncode == 0:
        raise consumer_err[0]
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg unified render failed with return code {proc.returncode}")

    elapsed = time.time() - t0
    render_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"[render] Rendered {frame_count} frames in {elapsed:.2f}s ({render_fps:.1f} fps) -> {out_p}")

    # Strict AV sync verification
    verify_av_sync(out_p, max_diff=0.05)
    return out_p

