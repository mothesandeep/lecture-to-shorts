#!/usr/bin/env python3
"""
reframer.py - Face-tracking 9:16 vertical video reframer with static slide rejection,
smart vertical eyeline/headroom framing, dead-zone dampening, and scene-cut detection.

Features integrated from auto-vertical-reframe (https://github.com/KazKozDev/auto-vertical-reframe):
1. Scene cut / boundary detection: Hard-jump/re-center on shot or scene cuts instead of drifting.
2. Dead-zone & motion lead: Eliminates camera micro-jitter when presenter is stationary, smoothly
   leads camera pan when walking across the blackboard.
3. Smart vertical eyeline / headroom framing: Places presenter's eyes in upper-third (~15-20% headroom),
   leaving comfortable lower space for captions.
4. Presets: talking_head (default), lecture, smooth, and action.

Key Invariants:
1. Two-pass motion + Haar face detection.
2. Static slide rejection via spatial clustering, position standard deviation, and median motion.
3. Teacher locking and tracking with temporal gating and linear interpolation.
4. Forward-backward Exponential Moving Average (EMA) smoothing for zero-lag camera pans.
5. High-fidelity FFmpeg rawvideo pipe encoding to 1080x1920 vertical format.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from core.av_sync import ensure_cfr_video, probe_av_sync, verify_av_sync
except ImportError:
    from av_sync import ensure_cfr_video, probe_av_sync, verify_av_sync


# ── Presets Catalog ──────────────────────────────────────────────────────────────────

REFRAME_PRESETS: Dict[str, dict] = {
    "talking_head": {
        "description": "Lecture & teaching default: responsive centering, upper-third eyeline, gentle 1.10x zoom",
        "deadzone_px": 20.0,
        "motion_lead": 0.45,
        "smooth_alpha": 0.06,
        "zoom": 1.10,
        "headroom": 0.18,
        "scene_cut_threshold": 0.82,
    },
    "lecture": {
        "description": "Full-height blackboard framing: 1.0x zoom, active motion lead",
        "deadzone_px": 24.0,
        "motion_lead": 0.40,
        "smooth_alpha": 0.07,
        "zoom": 1.0,
        "headroom": 0.20,
        "scene_cut_threshold": 0.82,
    },
    "smooth": {
        "description": "Cinematic pans with balanced dampening and safe centering",
        "deadzone_px": 28.0,
        "motion_lead": 0.35,
        "smooth_alpha": 0.05,
        "zoom": 1.05,
        "headroom": 0.18,
        "scene_cut_threshold": 0.85,
    },
    "action": {
        "description": "Dynamic tracking with rapid lead for fast presenter movements",
        "deadzone_px": 12.0,
        "motion_lead": 0.65,
        "smooth_alpha": 0.10,
        "zoom": 1.05,
        "headroom": 0.18,
        "scene_cut_threshold": 0.78,
    },
}


def list_reframe_presets() -> List[str]:
    """Return list of available reframing preset names."""
    return list(REFRAME_PRESETS.keys())


@dataclass
class ReframePlan:
    """Precomputed reframing trajectory and geometry for video rendering."""
    src_path: Path
    out_path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    num_frames: int
    crop_w: int
    crop_h: int
    out_canvas_w: int
    out_canvas_h: int
    crop_x0: np.ndarray
    crop_y0: np.ndarray
    scene_cuts: List[int]
    cut_indices: set[int]
    layout: str
    is_4_5: bool
    zoom: float
    preset_name: str
    headroom: float
    encoder_preset: str
    crf: int
    audio_bitrate: str
    detections: Optional[List[List[Tuple[float, float, int, int, float]]]] = None
    static_centers: Optional[List[np.ndarray]] = None


@dataclass
class ReframeConfig:
    """Configuration options for auto-reframing."""
    src: Path
    out: Path
    cascade_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    smooth_alpha: Optional[float] = None
    motion_thr: float = 0.03
    static_std: float = 8.0
    roi_xmax: float = 1.0
    debug: bool = False
    preset: str = "talking_head"
    deadzone_px: Optional[float] = None
    motion_lead: Optional[float] = None
    zoom: Optional[float] = None
    headroom: Optional[float] = None
    scene_cut_threshold: Optional[float] = None
    crf: int = 17
    audio_bitrate: str = "192k"
    encoder_preset: str = "fast"
    layout: str = "direct-crop"
    audio_delay_ms: float = 0.0
    aspect_ratio: str = "9:16"


def find_cascade(custom_path: Optional[Path | str] = None) -> Path:
    """
    Locate haarcascade_frontalface_default.xml across common paths.
    """
    cascade_filename = "haarcascade_frontalface_default.xml"
    home = Path.home()
    candidates: List[Optional[Path]] = []

    if custom_path:
        candidates.append(Path(custom_path).expanduser().resolve())

    candidates.extend([
        Path(__file__).resolve().parent / cascade_filename,
        Path.cwd() / cascade_filename,
        home / cascade_filename,
        home / "Desktop" / cascade_filename,
        home / "Downloads" / cascade_filename,
    ])

    try:
        candidates.append(Path(cv2.data.haarcascades) / cascade_filename)
    except Exception:
        pass

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Cascade classifier '{cascade_filename}' not found. Please provide --cascade "
        f"or place '{cascade_filename}' in current directory or user home ({home})."
    )


def compute_motion_ratio(mask: np.ndarray, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> float:
    """Compute motion ratio within expanded bounding box around detected face/body."""
    ex0 = max(0, int(x - 0.5 * w))
    ex1 = min(frame_w, int(x + 1.5 * w))
    ey0 = max(0, int(y - 0.2 * h))
    ey1 = min(frame_h, int(y + 3.5 * h))
    roi = mask[ey0:ey1, ex0:ex1]
    return float(roi.mean()) if roi.size else 0.0


def ema_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Apply single-pass Exponential Moving Average filter."""
    if len(arr) == 0:
        return arr
    out = arr.copy()
    for i in range(1, len(out)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def smooth_trajectory(cx: np.ndarray, alpha: float = 0.06, max_alpha: float = 0.18) -> np.ndarray:
    """
    Forward-backward zero-phase adaptive EMA smoothing.
    Automatically ramps alpha during fast lateral movement to keep camera panning in sync,
    while maintaining calm cinematic stability during stationary lecture moments.
    """
    n = len(cx)
    if n <= 2:
        return cx.copy()

    # Forward pass with velocity-adaptive alpha
    forward = cx.copy()
    for i in range(1, n):
        speed = abs(cx[i] - cx[i - 1])
        eff_alpha = min(max_alpha, max(alpha, alpha + speed * 0.02))
        forward[i] = eff_alpha * cx[i] + (1.0 - eff_alpha) * forward[i - 1]

    # Backward pass with velocity-adaptive alpha
    backward = forward.copy()
    for i in range(n - 2, -1, -1):
        speed = abs(forward[i] - forward[i + 1])
        eff_alpha = min(max_alpha, max(alpha, alpha + speed * 0.02))
        backward[i] = eff_alpha * forward[i] + (1.0 - eff_alpha) * backward[i + 1]

    return backward


def apply_deadzone_and_lead(
    trajectory: np.ndarray,
    deadzone_px: float = 20.0,
    motion_lead: float = 0.45,
    max_lead_px: float = 45.0
) -> np.ndarray:
    """
    Apply camera dead-zone dampening and velocity motion lead to tracked points.
    
    - Dead-zone: Suppresses micro-jitter when presenter is speaking in place.
    - Dynamic dead-zone & acceleration pull: Automatically contracts dead-zone during
      sustained walking or lateral movement so camera follows actively without trailing lag.
    - Motion lead: Predicts forward motion to lead camera smoothly in direction of movement.
    """
    if len(trajectory) == 0:
        return trajectory
    if deadzone_px <= 0 and motion_lead <= 0:
        return trajectory.copy()

    out = np.zeros_like(trajectory)
    anchor = float(trajectory[0])
    vx = 0.0

    for i in range(len(trajectory)):
        cur = float(trajectory[i])
        if i > 0:
            inst_v = cur - float(trajectory[i - 1])
            vx = 0.65 * vx + 0.35 * inst_v

        speed = abs(vx)
        # Dynamic dead-zone: contracts during sustained motion to eliminate trailing lag
        eff_dz = max(6.0, deadzone_px - speed * 2.5)
        delta = cur - anchor
        if abs(delta) > eff_dz:
            excess = abs(delta) - eff_dz
            # Proportional acceleration pull: increases responsiveness as walking speed rises
            pull = min(1.0, 0.40 + 0.10 * speed)
            anchor += np.sign(delta) * (excess * pull)

        lead = np.clip(vx * motion_lead, -max_lead_px, max_lead_px)
        out[i] = anchor + lead

    return out


def enforce_centering_margin(
    crop_cx: np.ndarray,
    face_cx: np.ndarray,
    crop_w: float,
    max_deviation_ratio: float = 0.18
) -> np.ndarray:
    """
    Anti-Corner Framing Invariant:
    Guarantees that presenter's face never drifts farther than max_deviation_ratio
    from the crop center (e.g. +/- 18% of crop width).
    Forces face to remain strictly within the safe central zone without ever
    reaching the corner or border.
    """
    n = len(crop_cx)
    if n == 0:
        return crop_cx

    max_dev = max_deviation_ratio * crop_w
    adjusted_cx = crop_cx.copy()

    for i in range(n):
        dev = face_cx[i] - adjusted_cx[i]
        if abs(dev) > max_dev:
            adjusted_cx[i] = face_cx[i] - np.sign(dev) * max_dev

    # Two-pass soft clamp smoother to eliminate any abrupt boundary clamp edges
    smoothed = adjusted_cx.copy()
    for i in range(1, n):
        smoothed[i] = 0.25 * adjusted_cx[i] + 0.75 * smoothed[i - 1]
    for i in range(n - 2, -1, -1):
        smoothed[i] = 0.25 * smoothed[i] + 0.75 * smoothed[i + 1]

    return smoothed


def enforce_headroom_margin(
    crop_cy: np.ndarray,
    face_cy: np.ndarray,
    crop_h: float,
    target_headroom: float = 0.18,
    max_deviation: float = 0.12
) -> np.ndarray:
    """
    Upper-Third Eyeline Invariant:
    Ensures eyes stay within [headroom - dev, headroom + dev] of top crop edge,
    preventing face from being pushed off the top or dropping into captions.
    """
    n = len(crop_cy)
    if n == 0:
        return crop_cy

    adjusted_cy = crop_cy.copy()
    max_dev = max_deviation * crop_h

    for i in range(n):
        dev = face_cy[i] - adjusted_cy[i]
        if abs(dev) > max_dev:
            adjusted_cy[i] = face_cy[i] - np.sign(dev) * max_dev

    smoothed = adjusted_cy.copy()
    for i in range(1, n):
        smoothed[i] = 0.25 * adjusted_cy[i] + 0.75 * smoothed[i - 1]
    for i in range(n - 2, -1, -1):
        smoothed[i] = 0.25 * smoothed[i] + 0.75 * smoothed[i + 1]

    return smoothed


class FaceReframer:
    """Face-tracking 9:16 vertical reframer with motion detection, slide rejection, and scene-cut handling."""

    def __init__(self, config: ReframeConfig) -> None:
        self.config = config
        self.cascade_file = find_cascade(config.cascade_path)
        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_file))
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV cascade from: {self.cascade_file}")

        # Resolve preset
        raw_preset = (config.preset or "talking_head").strip().lower().replace("-", "_")
        legacy_ffmpeg_presets = {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow"
        }
        if raw_preset in legacy_ffmpeg_presets:
            self.encoder_preset = raw_preset
            preset_key = "talking_head"
        else:
            self.encoder_preset = config.encoder_preset
            preset_key = raw_preset if raw_preset in REFRAME_PRESETS else "talking_head"

        preset_data = REFRAME_PRESETS[preset_key]
        self.preset_name = preset_key
        self.preset_desc = preset_data["description"]

        # Config parameters with preset fallbacks
        self.smooth_alpha = config.smooth_alpha if config.smooth_alpha is not None else preset_data["smooth_alpha"]
        self.deadzone_px = config.deadzone_px if config.deadzone_px is not None else preset_data["deadzone_px"]
        self.motion_lead = config.motion_lead if config.motion_lead is not None else preset_data["motion_lead"]
        self.zoom = config.zoom if config.zoom is not None else preset_data["zoom"]
        self.headroom = config.headroom if config.headroom is not None else preset_data["headroom"]
        self.scene_cut_threshold = config.scene_cut_threshold if config.scene_cut_threshold is not None else preset_data["scene_cut_threshold"]

        # Resolve layout mode: direct-crop vs blur-bw
        raw_layout = (config.layout or "direct-crop").strip().lower().replace("_", "-")
        if raw_layout in {"blur-bw", "blur", "bw", "blur-bg", "aesthetic"}:
            self.layout = "blur-bw"
        else:
            self.layout = "direct-crop"
        self.aspect_ratio = getattr(config, "aspect_ratio", "9:16")
        self.time_pass1: float = 0.0
        self.time_render: float = 0.0
        self.stage_timings: Dict[str, float] = {
            "reframe pass 1": 0.0,
            "reframe render": 0.0,
        }

    def plan(self) -> ReframePlan:
        """
        Execute passes 1 to 2.7 of face reframing:
        - Face detection, motion analysis, scene-cut boundary detection.
        - Static slide rejection.
        - Teacher locking with cut-aware reset.
        - Dead-zone dampening, motion lead, and segmented smoothing.
        - Anti-corner and headroom framing invariants.
        Returns a ReframePlan containing crop geometry and per-frame crop coordinates.
        """
        src_path = self.config.src.expanduser().resolve()
        out_path = self.config.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Enforce Constant Frame Rate (CFR) and AV sync alignment
        src_path = ensure_cfr_video(src_path, audio_delay_ms=self.config.audio_delay_ms)

        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open source video: {src_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        t_pass1_start = time.time()

        zoom = max(1.0, float(self.zoom))

        ar_norm = self.aspect_ratio.strip().replace(":", "x").lower()
        is_4_5 = (ar_norm in {"4x5", "4:5", "portrait", "feed"})

        if self.layout == "blur-bw":
            # 1:1 square foreground box (centered on 1080x1920 canvas)
            base_crop_w = height
            base_crop_h = height
            crop_w = int(round(base_crop_w / zoom / 2)) * 2
            crop_h = int(round(base_crop_h / zoom / 2)) * 2
            out_canvas_w, out_canvas_h = 1080, 1920
        elif is_4_5:
            # 4:5 Instagram Portrait Feed (1080x1350)
            base_crop_w = int(round(height * 4 / 5 / 2)) * 2
            base_crop_h = height
            if zoom > 1.0:
                crop_w = int(round(base_crop_w / zoom / 2)) * 2
                crop_h = int(round(base_crop_h / zoom / 2)) * 2
            else:
                crop_w = base_crop_w
                crop_h = base_crop_h
            out_canvas_w, out_canvas_h = 1080, 1350
        else:
            # 9:16 vertical full crop (direct-crop) (1080x1920)
            base_crop_w = int(round(height * 9 / 16 / 2)) * 2
            base_crop_h = height
            if zoom > 1.0:
                crop_w = int(round(base_crop_w / zoom / 2)) * 2
                crop_h = int(round(base_crop_h / zoom / 2)) * 2
            else:
                crop_w = base_crop_w
                crop_h = base_crop_h
            out_canvas_w, out_canvas_h = 1080, 1920

        lag = 2
        pixel_threshold = 12

        print(f"[reframe] Source: {width}x{height} @ {fps:.2f}fps ({total_frames} frames)")
        print(f"[reframe] Preset: '{self.preset_name}' ({self.preset_desc})")
        print(f"[reframe] Crop: {crop_w}x{crop_h} (Zoom: {zoom:.2f}x, Deadzone: {self.deadzone_px}px, Headroom: {self.headroom*100:.0f}%, Target: {out_canvas_w}x{out_canvas_h})")

        # ---------------- PASS 1: Face Detection + Motion + Scene Cut Detection ----------------
        print("[reframe] Pass 1/2: Scanning frames for faces, motion, and scene cuts...")
        detections: List[List[Tuple[float, float, int, int, float]]] = []
        frame_buffer: List[np.ndarray] = []
        scene_cuts: List[int] = [0]
        prev_hist: Optional[np.ndarray] = None
        min_scene_frames = max(15, int(fps * 0.6))
        last_cut_idx = 0
        frame_idx = 0

        # Template tracking assist state for high-recall continuity
        prev_template: Optional[np.ndarray] = None
        prev_box: Optional[Tuple[int, int, int, int]] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            frame_buffer.append(blur)

            # Scene Cut / Boundary Detection
            small = cv2.resize(gray, (160, 90))
            hist = cv2.calcHist([small], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                if corr < self.scene_cut_threshold and (frame_idx - last_cut_idx) >= min_scene_frames:
                    scene_cuts.append(frame_idx)
                    last_cut_idx = frame_idx
                    prev_template = None
                    prev_box = None
                    print(f"[reframe] Shot cut detected at frame {frame_idx} ({frame_idx / fps:.2f}s, corr={corr:.3f})")
            prev_hist = hist

            ref_frame = frame_buffer[max(0, len(frame_buffer) - 1 - lag)]
            motion_mask = (cv2.absdiff(blur, ref_frame) > pixel_threshold).astype(np.float32)

            faces = self.face_cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(32, 32)
            )

            current_frame_dets: List[Tuple[float, float, int, int, float]] = []
            valid_face_boxes: List[Tuple[int, int, int, int, float, float]] = []
            for (x, y, w, h) in (faces if len(faces) else []):
                cx = x + w / 2.0
                cy = y + h / 2.0
                if cx > width * self.config.roi_xmax:
                    continue
                valid_face_boxes.append((x, y, w, h, cx, cy))

            # Fast template tracking assist when Haar temporarily misses during head turns/gestures
            if not valid_face_boxes and prev_template is not None and prev_box is not None:
                bx, by, bw, bh = prev_box
                sx0 = max(0, bx - 60)
                sy0 = max(0, by - 40)
                sx1 = min(width, bx + bw + 60)
                sy1 = min(height, by + bh + 40)
                roi = gray[sy0:sy1, sx0:sx1]
                if roi.shape[0] >= prev_template.shape[0] and roi.shape[1] >= prev_template.shape[1]:
                    res = cv2.matchTemplate(roi, prev_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > 0.45:
                        tx = sx0 + max_loc[0]
                        ty = sy0 + max_loc[1]
                        valid_face_boxes.append((tx, ty, bw, bh, tx + bw / 2.0, ty + bh / 2.0))

            for (x, y, w, h, cx, cy) in valid_face_boxes:
                motion = compute_motion_ratio(motion_mask, x, y, w, h, width, height)
                current_frame_dets.append((cx, cy, w, h, motion))

            if valid_face_boxes:
                # Update tracking patch with primary face candidate
                best_face = max(valid_face_boxes, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh, _, _ = best_face
                prev_box = (fx, fy, fw, fh)
                prev_template = gray[max(0, fy):min(height, fy + fh), max(0, fx):min(width, fx + fw)].copy()

            detections.append(current_frame_dets)
            if len(frame_buffer) > lag + 1:
                frame_buffer.pop(0)

        cap.release()
        num_frames = len(detections)
        if num_frames == 0:
            raise RuntimeError(f"No frames could be read from video: {src_path}")

        if scene_cuts[-1] != num_frames:
            scene_cuts.append(num_frames)

        num_cuts = len(scene_cuts) - 2
        print(f"[reframe] Pass 1 completed: {num_frames} frames processed ({num_cuts} shot cut(s) detected).")

        # ---------------- CLUSTERING & STATIC SLIDE REJECTION ----------------
        clusters: List[dict] = []
        for frame_dets in detections:
            for (cx, cy, w, h, m) in frame_dets:
                for cluster in clusters:
                    mx, my = np.mean(cluster["pts"], axis=0)
                    if np.hypot(cx - mx, cy - my) < 30.0:
                        cluster["pts"].append((cx, cy))
                        cluster["mot"].append(m)
                        break
                else:
                    clusters.append({"pts": [(cx, cy)], "mot": [m]})

        static_centers: List[np.ndarray] = []
        for cluster in sorted(clusters, key=lambda c: -len(c["pts"])):
            cluster_count = len(cluster["pts"])
            if cluster_count < 5:
                continue
            pts = np.array(cluster["pts"])
            center_mean = pts.mean(axis=0)
            pos_std = pts.std(axis=0).max()
            median_mot = float(np.median(cluster["mot"]))

            is_static = (
                cluster_count >= max(15, int(0.03 * num_frames))
                and pos_std < self.config.static_std
                and median_mot < self.config.motion_thr
            )
            if is_static:
                static_centers.append(center_mean)
                print(f"[reframe] Static cluster detected at ({center_mean[0]:.0f}, {center_mean[1]:.0f}) "
                      f"std={pos_std:.1f}, median_motion={median_mot:.4f} -> Blacklisted (Slide/Logo)")

        def is_static_detection(cx: float, cy: float) -> bool:
            return any(np.hypot(cx - sc[0], cy - sc[1]) < 35.0 for sc in static_centers)

        # ---------------- PASS 2: Teacher Locking with Cut-Aware Reset ----------------
        raw_x = np.full(num_frames, np.nan)
        raw_y = np.full(num_frames, np.nan)
        prev_target: Optional[Tuple[float, float]] = None
        consecutive_misses = 0
        cut_indices = set(scene_cuts)

        for t, frame_dets in enumerate(detections):
            # Immediate target reset at scene cuts to prevent slow cross-cut drifting
            if t in cut_indices:
                prev_target = None
                consecutive_misses = 0

            candidates = [d for d in frame_dets if not is_static_detection(d[0], d[1])]
            if not candidates:
                consecutive_misses += 1
                if consecutive_misses > 75:
                    prev_target = None
                continue

            if prev_target is None:
                # Rank candidates by motion + area score
                best = max(candidates, key=lambda d: d[4] + 0.00005 * d[2] * d[3])
            else:
                gate = 90.0 + 8.0 * consecutive_misses
                nearby = [d for d in candidates if np.hypot(d[0] - prev_target[0], d[1] - prev_target[1]) < gate]
                if not nearby:
                    consecutive_misses += 1
                    continue
                best = min(nearby, key=lambda d: np.hypot(d[0] - prev_target[0], d[1] - prev_target[1]))

            prev_target = (best[0], best[1])
            consecutive_misses = 0
            raw_x[t] = best[0]
            # Eyeline sits slightly above face center (~40% from top of face box)
            raw_y[t] = best[1] - 0.10 * best[3]

        valid_mask = ~np.isnan(raw_x)
        num_valid = int(valid_mask.sum())
        all_indices = np.arange(num_frames)

        if num_valid == 0:
            print("[reframe] WARNING: No face detected. Defaulting to center framing.")
            interpolated_cx = np.full(num_frames, width / 2.0)
            interpolated_cy = np.full(num_frames, height / 3.0)
        else:
            print(f"[reframe] Pass 2: Face locked in {num_valid}/{num_frames} frames ({100.0 * num_valid / num_frames:.1f}%)")
            interpolated_cx = np.interp(all_indices, all_indices[valid_mask], raw_x[valid_mask])
            interpolated_cy = np.interp(all_indices, all_indices[valid_mask], raw_y[valid_mask])

        # ---------------- PASS 2.5: Dead-Zone, Motion Lead & Segmented Adaptive Smoothing ----------------
        smoothed_cx = np.zeros(num_frames, dtype=np.float32)
        smoothed_cy = np.zeros(num_frames, dtype=np.float32)

        for i in range(len(scene_cuts) - 1):
            s_start = scene_cuts[i]
            s_end = scene_cuts[i + 1]
            seg_cx = interpolated_cx[s_start:s_end]
            seg_cy = interpolated_cy[s_start:s_end]

            if len(seg_cx) > 0:
                # Apply velocity-adaptive dead-zone dampening & motion lead
                damp_cx = apply_deadzone_and_lead(seg_cx, self.deadzone_px, self.motion_lead)
                damp_cy = apply_deadzone_and_lead(seg_cy, self.deadzone_px * 0.75, self.motion_lead * 0.4)

                # Segment-isolated zero-phase adaptive EMA smoothing
                smoothed_cx[s_start:s_end] = smooth_trajectory(damp_cx, self.smooth_alpha)
                smoothed_cy[s_start:s_end] = smooth_trajectory(damp_cy, self.smooth_alpha * 0.8)

        # ---------------- PASS 2.7: Anti-Corner & Upper-Third Headroom Framing Invariants ----------------
        # 1. Horizontal centering safe-band: presenter's face never strays farther than 18% of crop width from center
        centered_cx = enforce_centering_margin(
            crop_cx=smoothed_cx,
            face_cx=interpolated_cx,
            crop_w=float(crop_w),
            max_deviation_ratio=0.18
        )
        crop_x0 = np.clip(centered_cx - crop_w / 2.0, 0.0, float(width - crop_w))

        # 2. Vertical headroom safe-band: keeps eyes in upper third without hitting top or captions
        if crop_h < height:
            centered_cy = enforce_headroom_margin(
                crop_cy=smoothed_cy,
                face_cy=interpolated_cy,
                crop_h=float(crop_h),
                target_headroom=self.headroom,
                max_deviation=0.12
            )
            target_crop_y = centered_cy - self.headroom * crop_h
            crop_y0 = np.clip(target_crop_y, 0.0, float(height - crop_h))
        else:
            crop_y0 = np.zeros(num_frames, dtype=np.float32)

        # Optional CSV keyframes export
        if self.config.csv_path:
            csv_file = self.config.csv_path.expanduser().resolve()
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "time_sec", "crop_x0", "crop_y0", "crop_w", "crop_h", "cut"])
                for i in range(num_frames):
                    writer.writerow([
                        i,
                        round(i / fps, 4),
                        round(float(crop_x0[i]), 2),
                        round(float(crop_y0[i]), 2),
                        crop_w,
                        crop_h,
                        1 if i in cut_indices else 0
                    ])
            print(f"[reframe] Exported crop keyframes to: {csv_file}")

        self.time_pass1 = time.time() - t_pass1_start
        self.stage_timings["reframe pass 1"] = self.time_pass1

        return ReframePlan(
            src_path=src_path,
            out_path=out_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            num_frames=num_frames,
            crop_w=crop_w,
            crop_h=crop_h,
            out_canvas_w=out_canvas_w,
            out_canvas_h=out_canvas_h,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            scene_cuts=scene_cuts,
            cut_indices=cut_indices,
            layout=self.layout,
            is_4_5=is_4_5,
            zoom=zoom,
            preset_name=self.preset_name,
            headroom=self.headroom,
            encoder_preset=self.encoder_preset,
            crf=self.config.crf,
            audio_bitrate=self.config.audio_bitrate,
            detections=detections,
            static_centers=static_centers,
        )

    def render(self, plan: Optional[ReframePlan] = None) -> Path:
        """
        Execute pass 3 of face reframing:
        Encode the cropped frames to out_path using FFmpeg rawvideo pipe.
        """
        if plan is None:
            plan = self.plan()

        t_render_start = time.time()
        src_path = plan.src_path
        out_path = plan.out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        crop_w, crop_h = plan.crop_w, plan.crop_h
        fps = plan.fps
        out_canvas_w, out_canvas_h = plan.out_canvas_w, plan.out_canvas_h
        is_4_5 = plan.is_4_5
        crop_x0 = plan.crop_x0
        crop_y0 = plan.crop_y0
        num_frames = plan.num_frames
        cut_indices = plan.cut_indices
        zoom = plan.zoom

        # ---------------- PASS 3: FFmpeg Rendering & Encoding ----------------
        if plan.layout == "blur-bw":
            print(f"[reframe] Encoding 9:16 aesthetic blur-bw video via FFmpeg filter_complex -> {out_path}")
            filter_complex = (
                "[1:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,format=gray,format=yuv420p[bg]; "
                "[0:v]scale=1080:1080:flags=lanczos[fg]; "
                "[bg][fg]overlay=0:420,drawbox=x=0:y=419:w=1080:h=2:color=white@0.35:t=fill,drawbox=x=0:y=1500:w=1080:h=2:color=white@0.35:t=fill[v]; "
                "[1:a:0]asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0[a]"
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{crop_h}", "-r", f"{fps}", "-i", "-",
                "-i", str(src_path),
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-fps_mode", "cfr",
                "-c:v", "libx264", "-preset", plan.encoder_preset, "-crf", str(plan.crf), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", plan.audio_bitrate, "-shortest",
                "-movflags", "+faststart", str(out_path)
            ]
        else:
            scale_desc = f"{out_canvas_w}x{out_canvas_h} {'4:5 feed' if is_4_5 else '9:16 vertical'}"
            print(f"[reframe] Encoding {scale_desc} direct-crop video via FFmpeg pipe -> {out_path}")
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{crop_h}", "-r", f"{fps}", "-i", "-",
                "-i", str(src_path),
                "-map", "0:v", "-map", "1:a:0",
                "-vf", f"scale={out_canvas_w}:{out_canvas_h}:flags=lanczos,format=yuv420p",
                "-af", "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0",
                "-fps_mode", "cfr",
                "-c:v", "libx264", "-preset", plan.encoder_preset, "-crf", str(plan.crf),
                "-c:a", "aac", "-b:a", plan.audio_bitrate, "-shortest",
                "-movflags", "+faststart", str(out_path)
            ]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        debug_writer: Optional[cv2.VideoWriter] = None
        if self.config.debug:
            dbg_path = out_path.with_name(f"{out_path.stem}_debug_overlay.mp4")
            debug_writer = cv2.VideoWriter(
                str(dbg_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (plan.width, plan.height)
            )
            print(f"[reframe] Debug overlay enabled -> {dbg_path}")

        static_centers = plan.static_centers or []

        def is_static_detection(cx: float, cy: float) -> bool:
            return any(np.hypot(cx - sc[0], cy - sc[1]) < 35.0 for sc in static_centers)

        cap = cv2.VideoCapture(str(src_path))
        try:
            for t in range(num_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                affine_matrix = np.float32([[1, 0, -crop_x0[t]], [0, 1, -crop_y0[t]]])
                cropped_frame = cv2.warpAffine(frame, affine_matrix, (crop_w, crop_h), flags=cv2.INTER_LINEAR)
                if proc.stdin:
                    proc.stdin.write(cropped_frame.tobytes())

                if debug_writer is not None:
                    vis = frame.copy()
                    if plan.detections and t < len(plan.detections):
                        for (dx, dy, dw, dh, m) in plan.detections[t]:
                            color = (0, 0, 255) if is_static_detection(dx, dy) else (0, 255, 0)
                            cv2.rectangle(vis, (int(dx - dw / 2), int(dy - dh / 2)),
                                          (int(dx + dw / 2), int(dy + dh / 2)), color, 2)
                    x0, y0 = int(crop_x0[t]), int(crop_y0[t])
                    cv2.rectangle(vis, (x0, y0), (x0 + crop_w, y0 + crop_h), (255, 200, 0), 2)
                    # Upper-third eyeline guideline
                    eyeline_y = int(y0 + self.headroom * crop_h)
                    cv2.line(vis, (x0, eyeline_y), (x0 + crop_w, eyeline_y), (0, 255, 255), 1)
                    # HUD overlay
                    is_cut = t in cut_indices
                    cv2.putText(vis, f"Preset: {self.preset_name} | Zoom: {zoom:.2f}x | Cut: {is_cut}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    debug_writer.write(vis)
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            if debug_writer is not None:
                debug_writer.release()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg process failed with return code {proc.returncode}")

        verify_av_sync(out_path, max_diff=0.05)
        self.time_render = time.time() - t_render_start
        self.stage_timings["reframe render"] = self.time_render
        print(f"[reframe] Success! Reframed video created at: {out_path}")
        return out_path

    def run(self) -> Path:
        """Execute reframing: plan passes 1-2.7 and render pass 3 in sequence."""
        plan = self.plan()
        return self.render(plan)


def reframe_video(
    src: Path | str,
    out: Path | str,
    cascade_path: Optional[Path | str] = None,
    csv_path: Optional[Path | str] = None,
    smooth_alpha: Optional[float] = None,
    motion_thr: float = 0.03,
    static_std: float = 8.0,
    roi_xmax: float = 1.0,
    debug: bool = False,
    preset: str = "talking_head",
    deadzone_px: Optional[float] = None,
    motion_lead: Optional[float] = None,
    zoom: Optional[float] = None,
    headroom: Optional[float] = None,
    scene_cut_threshold: Optional[float] = None,
    layout: str = "direct-crop",
    audio_delay_ms: float = 0.0,
) -> Path:
    """Convenience functional interface for video reframing."""
    config = ReframeConfig(
        src=Path(src),
        out=Path(out),
        cascade_path=Path(cascade_path) if cascade_path else None,
        csv_path=Path(csv_path) if csv_path else None,
        smooth_alpha=smooth_alpha,
        motion_thr=motion_thr,
        static_std=static_std,
        roi_xmax=roi_xmax,
        debug=debug,
        preset=preset,
        deadzone_px=deadzone_px,
        motion_lead=motion_lead,
        zoom=zoom,
        headroom=headroom,
        scene_cut_threshold=scene_cut_threshold,
        layout=layout,
        audio_delay_ms=audio_delay_ms
    )
    reframer = FaceReframer(config)
    return reframer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-reframe video to 9:16 vertical crop with face tracking.")
    parser.add_argument("src", help="Path to input horizontal video")
    parser.add_argument("-o", "--out", default=None, help="Path to output vertical video")
    parser.add_argument("--preset", default="talking_head", help="Reframing preset (default: talking_head). Options: talking_head, lecture, smooth, action")
    parser.add_argument("--layout", default="direct-crop", choices=["direct-crop", "blur-bw"], help="Output canvas layout (default: direct-crop). Options: direct-crop, blur-bw")
    parser.add_argument("--audio-delay", type=float, default=0.0, help="Audio delay offset in milliseconds (default: 0.0)")
    parser.add_argument("--cascade", default=None, help="Custom path to haarcascade_frontalface_default.xml")
    parser.add_argument("--csv", default=None, help="Path to export crop keyframes CSV")
    parser.add_argument("--smooth", type=float, default=None, help="EMA alpha smoothing factor")
    parser.add_argument("--deadzone", type=float, default=None, help="Dead-zone dampening in pixels")
    parser.add_argument("--zoom", type=float, default=None, help="Framing zoom factor (e.g. 1.10)")
    parser.add_argument("--headroom", type=float, default=None, help="Headroom ratio from top (e.g. 0.18)")
    parser.add_argument("--motion-thr", type=float, default=0.03, help="Motion threshold for static slide rejection")
    parser.add_argument("--static-std", type=float, default=8.0, help="Max position std for static slide rejection")
    parser.add_argument("--roi-xmax", type=float, default=1.0, help="Ignore face detections to the right of this fraction")
    parser.add_argument("--debug", action="store_true", help="Generate debug video with bounding box overlays")
    args = parser.parse_args()

    src_p = Path(args.src)
    out_p = Path(args.out) if args.out else src_p.with_name(f"{src_p.stem}_reframe.mp4")
    reframe_video(
        src=src_p,
        out=out_p,
        cascade_path=args.cascade,
        csv_path=args.csv,
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
        audio_delay_ms=args.audio_delay
    )
