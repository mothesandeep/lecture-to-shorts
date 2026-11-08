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
    from core.audio_features import AudioFeatureExtractor
except ImportError:
    try:
        from av_sync import ensure_cfr_video, probe_av_sync, verify_av_sync
        from audio_features import AudioFeatureExtractor
    except ImportError:
        AudioFeatureExtractor = None

try:
    from scripts.download_models import ensure_yunet_model
except ImportError:
    try:
        from download_models import ensure_yunet_model
    except ImportError:
        ensure_yunet_model = None

try:
    import Vision
    import Foundation
    _VISION_AVAILABLE = True
except Exception:
    _VISION_AVAILABLE = False


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
    lock_pct: float = 0.0
    num_valid_faces: int = 0
    detect_every: int = 3
    crop_w_arr: Optional[np.ndarray] = None
    crop_h_arr: Optional[np.ndarray] = None
    zoom_curve: Optional[np.ndarray] = None
    detector: str = "haar"
    face_lock_count: int = 0
    motion_fallback_count: int = 0
    hold_count: int = 0


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
    detect_every: int = 3
    detect_width: int = 320
    punch_in: str = "off"
    punch_in_events: Optional[List[Tuple[float, float]]] = None
    words: Optional[List] = None
    detector: str = "haar"
    model_path: Optional[Path | str] = None


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


# ── Pluggable Face Detectors ──────────────────────────────────────────────────────────

class BaseFaceDetector:
    """Base interface for pluggable vertical reframer face detectors."""
    name: str = "base"

    def detect(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int, float, float]]:
        """
        Detect face bounding boxes in frame.
        Returns list of (x, y, w, h, cx, cy) in original frame pixel coordinates.
        """
        raise NotImplementedError


class HaarFaceDetector(BaseFaceDetector):
    """OpenCV Haar Cascade Face Detector (CPU-based, robust baseline)."""
    name: str = "haar"

    def __init__(self, cascade_path: Path | str, detect_width: int = 320) -> None:
        self.cascade_path = Path(cascade_path).expanduser().resolve()
        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV cascade from: {self.cascade_path}")
        self.detect_width = detect_width

    def detect(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int, float, float]]:
        height, width = frame.shape[:2]
        if width > self.detect_width:
            scale = self.detect_width / width
            det_w = self.detect_width
            det_h = max(1, int(round(height * scale)))
            inv_scale_x = width / det_w
            inv_scale_y = height / det_h
            min_size = max(24, int(round(32 * scale)))
            det_gray = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_gray = gray
            inv_scale_x, inv_scale_y = 1.0, 1.0
            min_size = 32

        faces = self.face_cascade.detectMultiScale(
            det_gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size)
        )
        boxes: List[Tuple[int, int, int, int, float, float]] = []
        for (sx, sy, sw, sh) in (faces if len(faces) else []):
            x = int(round(sx * inv_scale_x))
            y = int(round(sy * inv_scale_y))
            w = int(round(sw * inv_scale_x))
            h = int(round(sh * inv_scale_y))
            cx = x + w / 2.0
            cy = y + h / 2.0
            boxes.append((x, y, w, h, cx, cy))
        return boxes


class YuNetFaceDetector(BaseFaceDetector):
    """OpenCV YuNet ONNX Face Detector (neural network, high recall & fast)."""
    name: str = "yunet"

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        detect_width: int = 320,
        score_threshold: float = 0.6
    ) -> None:
        self.detect_width = detect_width
        self.score_threshold = score_threshold

        resolved_path: Optional[Path] = None
        if model_path is not None:
            candidate = Path(model_path).expanduser().resolve()
            if candidate.is_file():
                resolved_path = candidate
            else:
                raise FileNotFoundError(f"YuNet model file not found at specified path: {model_path}")

        if resolved_path is None:
            # Check local models directories
            candidates = [
                Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx",
                Path.cwd() / "models" / "face_detection_yunet_2023mar.onnx",
                Path.cwd() / "core" / "face_detection_yunet_2023mar.onnx",
                Path.home() / ".cache" / "models" / "face_detection_yunet_2023mar.onnx",
            ]
            for c in candidates:
                if c.is_file() and c.stat().st_size > 100000:
                    resolved_path = c
                    break

        if resolved_path is None and ensure_yunet_model is not None:
            try:
                resolved_path = ensure_yunet_model()
            except Exception as exc:
                print(f"[reframe] Warning: Failed to auto-download YuNet model: {exc}")

        if resolved_path is None or not resolved_path.is_file():
            raise FileNotFoundError(
                "YuNet ONNX model not found. Run 'python3 scripts/download_models.py' or specify --model-path."
            )

        self.model_path = resolved_path
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        if self.detector is None:
            raise RuntimeError(f"Failed to initialize cv2.FaceDetectorYN with model: {self.model_path}")
        self._cur_input_size = (320, 320)

    def detect(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int, float, float]]:
        height, width = frame.shape[:2]
        if width > self.detect_width:
            scale = self.detect_width / width
            det_w = self.detect_width
            det_h = max(1, int(round(height * scale)))
            inv_scale_x = width / det_w
            inv_scale_y = height / det_h
            det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_frame = frame
            det_w, det_h = width, height
            inv_scale_x, inv_scale_y = 1.0, 1.0

        if (det_w, det_h) != self._cur_input_size:
            self.detector.setInputSize((det_w, det_h))
            self._cur_input_size = (det_w, det_h)

        _, detections = self.detector.detect(det_frame)
        boxes: List[Tuple[int, int, int, int, float, float]] = []
        if detections is not None:
            for face in detections:
                fx, fy, fw, fh = face[:4]
                x = int(round(fx * inv_scale_x))
                y = int(round(fy * inv_scale_y))
                w = int(round(fw * inv_scale_x))
                h = int(round(fh * inv_scale_y))
                cx = x + w / 2.0
                cy = y + h / 2.0
                boxes.append((x, y, w, h, cx, cy))
        return boxes


class VisionFaceDetector(BaseFaceDetector):
    """Apple Vision framework Face Detector via PyObjC."""
    name: str = "vision"

    def __init__(self, detect_width: int = 320) -> None:
        self.detect_width = detect_width
        if not _VISION_AVAILABLE:
            raise RuntimeError(
                "Apple Vision framework via PyObjC is not available. "
                "Ensure pyobjc-framework-Vision is installed or use --detector haar."
            )
        import Vision  # type: ignore
        import Foundation  # type: ignore
        self._Vision = Vision
        self._Foundation = Foundation
        self._face_req = Vision.VNDetectFaceRectanglesRequest.alloc().init()

    def detect(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int, float, float]]:
        height, width = frame.shape[:2]
        if width > self.detect_width:
            scale = self.detect_width / width
            det_w = self.detect_width
            det_h = max(1, int(round(height * scale)))
            det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_frame = frame

        ok, buf = cv2.imencode(".jpg", det_frame)
        if not ok:
            return []

        raw_bytes = buf.tobytes()
        nsdata = self._Foundation.NSData.dataWithBytes_length_(raw_bytes, len(raw_bytes))
        handler = self._Vision.VNImageRequestHandler.alloc().initWithData_options_(nsdata, None)
        success, _ = handler.performRequests_error_([self._face_req], None)
        if not success:
            return []

        results = self._face_req.results() or []
        boxes: List[Tuple[int, int, int, int, float, float]] = []
        for face in results:
            bb = face.boundingBox()
            x = int(round(bb.origin.x * width))
            y = int(round((1.0 - bb.origin.y - bb.size.height) * height))
            w = int(round(bb.size.width * width))
            h = int(round(bb.size.height * height))
            cx = x + w / 2.0
            cy = y + h / 2.0
            boxes.append((x, y, w, h, cx, cy))
        return boxes


def create_face_detector(
    detector_type: str = "haar",
    cascade_path: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
    detect_width: int = 320,
) -> BaseFaceDetector:
    """
    Factory function for pluggable vertical reframing face detectors:
    - 'haar'   : OpenCV Haar Cascade (default)
    - 'yunet'  : OpenCV YuNet ONNX
    - 'vision' : Apple Vision framework via PyObjC
    Falls back gracefully to Haar if the requested detector cannot be initialized.
    """
    det_lower = (detector_type or "haar").strip().lower()

    if det_lower == "yunet":
        try:
            return YuNetFaceDetector(model_path=model_path, detect_width=detect_width)
        except Exception as exc:
            print(f"[reframe] WARNING: YuNet detector failed to initialize ({exc}). Falling back to Haar cascade.")
            cascade_file = find_cascade(cascade_path)
            return HaarFaceDetector(cascade_file, detect_width=detect_width)

    elif det_lower == "vision":
        try:
            return VisionFaceDetector(detect_width=detect_width)
        except Exception as exc:
            print(f"[reframe] WARNING: Vision detector unavailable ({exc}). Falling back to Haar cascade.")
            cascade_file = find_cascade(cascade_path)
            return HaarFaceDetector(cascade_file, detect_width=detect_width)

    else:
        # Default: haar
        cascade_file = find_cascade(cascade_path)
        return HaarFaceDetector(cascade_file, detect_width=detect_width)


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


def smoothstep(edge0: float, edge1: float, x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard smoothstep ease-in / ease-out polynomial: S(u) = 3*u^2 - 2*u^3.
    Maps x in [edge0, edge1] to [0.0, 1.0] with zero 1st derivatives at boundaries.
    """
    if edge0 == edge1:
        if isinstance(x, np.ndarray):
            return np.where(x < edge0, 0.0, 1.0)
        return 0.0 if x < edge0 else 1.0
    u = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def zoom_curve(
    n: int,
    fps: float,
    events: List[Tuple[float, float]],
    z_max: float = 1.20,
    ramp: float = 0.35,
) -> np.ndarray:
    """
    Generate per-frame punch-in zoom trajectory using smoothstep ease-in / ease-out.
    
    Args:
        n: Total frame count.
        fps: Frames per second.
        events: List of (start_time_s, end_time_s) zoom intervals.
        z_max: Peak punch-in zoom factor (e.g. 1.12x for <=480p, 1.25x for >=720p).
        ramp: Transition ramp duration in seconds for ease-in and ease-out.
        
    Returns:
        np.ndarray of shape (n,) with zoom multipliers in [1.0, z_max].
    """
    if n <= 0:
        return np.array([], dtype=np.float32)
    if fps <= 0:
        fps = 25.0
    if z_max <= 1.0 or not events:
        return np.ones(n, dtype=np.float32)

    times = np.arange(n, dtype=np.float32) / float(fps)
    curve = np.ones(n, dtype=np.float32)
    ramp_s = max(0.01, float(ramp))

    for (start_t, end_t) in events:
        start_t = float(start_t)
        end_t = float(end_t)
        if end_t <= start_t:
            continue
        dur = end_t - start_t
        eff_ramp = min(ramp_s, dur / 2.0)

        t_ramp_in_end = start_t + eff_ramp
        t_ramp_out_start = end_t - eff_ramp

        event_profile = np.zeros(n, dtype=np.float32)

        mask_ramp_in = (times >= start_t) & (times < t_ramp_in_end)
        mask_hold = (times >= t_ramp_in_end) & (times <= t_ramp_out_start)
        mask_ramp_out = (times > t_ramp_out_start) & (times <= end_t)

        if np.any(mask_ramp_in):
            event_profile[mask_ramp_in] = smoothstep(start_t, t_ramp_in_end, times[mask_ramp_in])
        if np.any(mask_hold):
            event_profile[mask_hold] = 1.0
        if np.any(mask_ramp_out):
            event_profile[mask_ramp_out] = 1.0 - smoothstep(t_ramp_out_start, end_t, times[mask_ramp_out])

        event_zoom = 1.0 + (z_max - 1.0) * event_profile
        curve = np.maximum(curve, event_zoom)

    return np.clip(curve, 1.0, z_max).astype(np.float32)


def get_max_punch_in_zoom(height: int) -> float:
    """
    Cap dynamic punch-in zoom level by source video resolution to avoid blur / pixelation:
    - 480p or lower (<= 480px height): cap at 1.12x.
    - 720p or higher (>= 720px height): cap at 1.25x.
    - Between 480p and 720p: linear interpolation.
    """
    if height <= 480:
        return 1.12
    elif height >= 720:
        return 1.25
    else:
        ratio = (height - 480) / (720 - 480)
        return round(1.12 + ratio * (1.25 - 1.12), 3)


def detect_punch_in_events(
    media_path: Optional[Path | str] = None,
    words: Optional[List] = None,
    min_gap_s: float = 3.0,
    peak_z_threshold: float = 1.2,
    hold_s: float = 1.4,
) -> List[Tuple[float, float]]:
    """
    Detect punch-in zoom timestamps from:
    1. Audio RMS energy peaks (AudioFeatureExtractor z >= 1.2).
    2. Emphasis words (!, ?, ALL-CAPS >= 3 chars, or duration >= 0.45s).
    
    Enforces minimum spacing between events to prevent visual fatigue.
    """
    candidates: List[Tuple[float, float, float]] = []

    if media_path and AudioFeatureExtractor is not None:
        p = Path(media_path).expanduser().resolve()
        if p.is_file():
            try:
                extractor = AudioFeatureExtractor.from_file(
                    p,
                    window_duration_s=2.0,
                    hop_duration_s=1.0,
                )
                if len(extractor.window_centers) > 0 and len(extractor.energy_zscores) > 0:
                    peak_mask = extractor.energy_zscores >= peak_z_threshold
                    for center_t, z in zip(extractor.window_centers[peak_mask], extractor.energy_zscores[peak_mask]):
                        t_c = float(center_t)
                        start_t = max(0.5, t_c - 0.2)
                        end_t = start_t + hold_s
                        candidates.append((start_t, end_t, float(z)))
            except Exception:
                pass

    if words:
        for w in words:
            w_start = getattr(w, "start", None) if hasattr(w, "start") else (w.get("start") if isinstance(w, dict) else None)
            w_end = getattr(w, "end", None) if hasattr(w, "end") else (w.get("end") if isinstance(w, dict) else None)
            w_text = getattr(w, "text", "") if hasattr(w, "text") else (w.get("text", "") if isinstance(w, dict) else "")
            if w_start is None or w_end is None:
                continue
            w_start = float(w_start)
            w_end = float(w_end)
            txt = str(w_text).strip()
            dur = w_end - w_start

            is_emp = False
            score = 1.0
            if "!" in txt or "?" in txt:
                is_emp = True
                score = 2.0
            elif txt.isupper() and len(txt) >= 3 and not txt.isdigit():
                is_emp = True
                score = 1.5
            elif dur >= 0.45:
                is_emp = True
                score = 1.2

            if is_emp and w_start >= 0.5:
                start_t = max(0.5, w_start - 0.1)
                end_t = start_t + max(hold_s, dur + 0.4)
                candidates.append((start_t, end_t, score))

    if not candidates:
        return []

    candidates.sort(key=lambda c: c[0])
    clusters: List[List[Tuple[float, float, float]]] = []
    current_cluster: List[Tuple[float, float, float]] = [candidates[0]]

    for cand in candidates[1:]:
        if cand[0] - current_cluster[-1][0] < min_gap_s:
            current_cluster.append(cand)
        else:
            clusters.append(current_cluster)
            current_cluster = [cand]
    if current_cluster:
        clusters.append(current_cluster)

    selected: List[Tuple[float, float]] = []
    last_end = -min_gap_s
    for cluster in clusters:
        best = max(cluster, key=lambda c: c[2])
        if best[0] - last_end >= min_gap_s * 0.5:
            selected.append((round(best[0], 2), round(best[1], 2)))
            last_end = best[1]

    return selected


def enforce_centering_margin(
    crop_cx: np.ndarray,
    face_cx: np.ndarray,
    crop_w: float | np.ndarray,
    max_deviation_ratio: float = 0.18
) -> np.ndarray:
    """
    Anti-Corner Framing Invariant:
    Guarantees that presenter's face never drifts farther than max_deviation_ratio
    from the crop center (e.g. +/- 18% of crop width).
    Forces face to remain strictly within the safe central zone [0.32, 0.68] of crop width
    without ever reaching the corner or border.
    """
    n = len(crop_cx)
    if n == 0:
        return crop_cx

    max_dev = max_deviation_ratio * crop_w
    is_arr = isinstance(max_dev, np.ndarray) and len(max_dev) == n
    adjusted_cx = crop_cx.copy()

    for i in range(n):
        lim = float(max_dev[i]) if is_arr else float(max_dev)
        dev = face_cx[i] - adjusted_cx[i]
        if abs(dev) > lim:
            adjusted_cx[i] = face_cx[i] - np.sign(dev) * lim

    # Two-pass soft clamp smoother to eliminate any abrupt boundary clamp edges
    smoothed = adjusted_cx.copy()
    for i in range(1, n):
        smoothed[i] = 0.25 * adjusted_cx[i] + 0.75 * smoothed[i - 1]
    for i in range(n - 2, -1, -1):
        smoothed[i] = 0.25 * smoothed[i] + 0.75 * smoothed[i + 1]

    # Final guarantee clamp against zoomed crop width to enforce invariant strictly
    for i in range(n):
        lim = float(max_dev[i]) if is_arr else float(max_dev)
        dev = face_cx[i] - smoothed[i]
        if abs(dev) > lim:
            smoothed[i] = face_cx[i] - np.sign(dev) * lim

    return smoothed


def enforce_headroom_margin(
    crop_cy: np.ndarray,
    face_cy: np.ndarray,
    crop_h: float | np.ndarray,
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
    is_arr = isinstance(max_dev, np.ndarray) and len(max_dev) == n

    for i in range(n):
        lim = float(max_dev[i]) if is_arr else float(max_dev)
        dev = face_cy[i] - adjusted_cy[i]
        if abs(dev) > lim:
            adjusted_cy[i] = face_cy[i] - np.sign(dev) * lim

    smoothed = adjusted_cy.copy()
    for i in range(1, n):
        smoothed[i] = 0.25 * adjusted_cy[i] + 0.75 * smoothed[i - 1]
    for i in range(n - 2, -1, -1):
        smoothed[i] = 0.25 * smoothed[i] + 0.75 * smoothed[i + 1]

    # Final guarantee clamp against zoomed crop height
    for i in range(n):
        lim = float(max_dev[i]) if is_arr else float(max_dev)
        dev = face_cy[i] - smoothed[i]
        if abs(dev) > lim:
            smoothed[i] = face_cy[i] - np.sign(dev) * lim

    return smoothed


class FaceReframer:
    """Face-tracking 9:16 vertical reframer with motion detection, slide rejection, and scene-cut handling."""

    def __init__(self, config: ReframeConfig) -> None:
        self.config = config
        self.cascade_file = find_cascade(config.cascade_path)
        try:
            self.face_cascade = cv2.CascadeClassifier(str(self.cascade_file))
        except Exception:
            self.face_cascade = None

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
        self.detect_every = max(1, getattr(config, "detect_every", 3))
        self.detect_width = max(160, getattr(config, "detect_width", 320))
        self.punch_in = getattr(config, "punch_in", "off")

        # Initialize pluggable detector (haar | yunet | vision)
        self.detector_name = getattr(config, "detector", "haar")
        self.model_path = getattr(config, "model_path", None)
        self.detector = create_face_detector(
            detector_type=self.detector_name,
            cascade_path=self.cascade_file,
            model_path=self.model_path,
            detect_width=self.detect_width,
        )

        self.time_pass1: float = 0.0
        self.time_render: float = 0.0
        self.stage_timings: Dict[str, float] = {
            "reframe pass 1": 0.0,
            "reframe render": 0.0,
        }

    def plan(self, words: Optional[List] = None) -> ReframePlan:
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
        print(f"[reframe] Pass 1/2: Scanning frames for faces (detector: {self.detector.name}), motion, and scene cuts...")
        detections: List[List[Tuple[float, float, int, int, float]]] = []
        motion_blobs: List[List[Tuple[float, float, float]]] = []
        frame_buffer: List[np.ndarray] = []
        scene_cuts: List[int] = [0]
        prev_hist: Optional[np.ndarray] = None
        min_scene_frames = max(15, int(fps * 0.6))
        last_cut_idx = 0
        frame_idx = 0

        detect_every = self.detect_every

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
            diff = cv2.absdiff(blur, ref_frame)
            motion_mask = (diff > pixel_threshold).astype(np.float32)

            # Extract motion contours for back-turned fallback
            motion_uint8 = (diff > pixel_threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(motion_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_motion_blobs: List[Tuple[float, float, float]] = []
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area >= 120.0:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        mcx = float(M["m10"] / M["m00"])
                        mcy = float(M["m01"] / M["m00"])
                        frame_motion_blobs.append((mcx, mcy, area))
            motion_blobs.append(frame_motion_blobs)

            valid_face_boxes: List[Tuple[int, int, int, int, float, float]] = []

            # 1. Run face detector every Nth frame (or if face tracking lost)
            run_detector = (detect_every <= 1) or ((frame_idx - 1) % detect_every == 0) or (prev_template is None)
            if run_detector:
                raw_boxes = self.detector.detect(frame, gray)
                for (x, y, w, h, cx, cy) in raw_boxes:
                    if cx > width * self.config.roi_xmax:
                        continue
                    valid_face_boxes.append((x, y, w, h, cx, cy))

            # 2. Fast template tracking assist when detector wasn't run or temporarily misses
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

            current_frame_dets: List[Tuple[float, float, int, int, float]] = []
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

        # ---------------- PASS 2: Teacher Locking with Back-Turned Fallback & Cut-Aware Reset ----------------
        raw_x = np.full(num_frames, np.nan)
        raw_y = np.full(num_frames, np.nan)
        prev_target: Optional[Tuple[float, float]] = None
        consecutive_misses = 0
        cut_indices = set(scene_cuts)
        face_lock_count = 0
        motion_fallback_count = 0
        hold_count = 0

        for t, frame_dets in enumerate(detections):
            # Immediate target reset at scene cuts to prevent slow cross-cut drifting
            if t in cut_indices:
                prev_target = None
                consecutive_misses = 0

            candidates = [d for d in frame_dets if not is_static_detection(d[0], d[1])]
            best_face = None
            if candidates:
                if prev_target is None:
                    # Rank candidates by motion + area score
                    best_face = max(candidates, key=lambda d: d[4] + 0.00005 * d[2] * d[3])
                else:
                    gate = 90.0 + 8.0 * min(consecutive_misses, 15)
                    nearby = [d for d in candidates if np.hypot(d[0] - prev_target[0], d[1] - prev_target[1]) < gate]
                    if nearby:
                        best_face = min(nearby, key=lambda d: np.hypot(d[0] - prev_target[0], d[1] - prev_target[1]))

            if best_face is not None:
                prev_target = (best_face[0], best_face[1])
                consecutive_misses = 0
                raw_x[t] = best_face[0]
                # Eyeline sits slightly above face center (~40% from top of face box)
                raw_y[t] = best_face[1] - 0.10 * best_face[3]
                face_lock_count += 1
            else:
                consecutive_misses += 1
                if prev_target is not None:
                    if consecutive_misses >= 3:
                        # Back-turned fallback: search for motion-mask centroid near last known position
                        motion_gate = 140.0 + 10.0 * min(consecutive_misses, 15)
                        frame_motion = motion_blobs[t] if t < len(motion_blobs) else []
                        nearby_motion = [
                            m for m in frame_motion
                            if np.hypot(m[0] - prev_target[0], m[1] - prev_target[1]) < motion_gate
                        ]
                        if nearby_motion:
                            # Pick prominent motion blob near previous position
                            best_mot = max(
                                nearby_motion,
                                key=lambda m: m[2] / (1.0 + 0.05 * np.hypot(m[0] - prev_target[0], m[1] - prev_target[1]))
                            )
                            raw_x[t] = best_mot[0]
                            raw_y[t] = prev_target[1]
                            prev_target = (0.75 * prev_target[0] + 0.25 * best_mot[0], prev_target[1])
                            motion_fallback_count += 1
                        else:
                            # If motion also fails: HOLD position
                            raw_x[t] = prev_target[0]
                            raw_y[t] = prev_target[1]
                            hold_count += 1
                    else:
                        # Transient miss (< 3 frames): HOLD position
                        raw_x[t] = prev_target[0]
                        raw_y[t] = prev_target[1]
                        hold_count += 1

                if consecutive_misses > 75:
                    prev_target = None

        valid_mask = ~np.isnan(raw_x)
        num_valid = int(valid_mask.sum())
        face_lock_pct = 100.0 * face_lock_count / num_frames if num_frames > 0 else 0.0
        all_indices = np.arange(num_frames)

        if num_valid == 0:
            print("[reframe] WARNING: No face detected. Defaulting to center framing.")
            interpolated_cx = np.full(num_frames, width / 2.0)
            interpolated_cy = np.full(num_frames, height / 3.0)
        else:
            print(
                f"[reframe] Pass 2: Face locked in {face_lock_count}/{num_frames} frames ({face_lock_pct:.1f}%), "
                f"motion fallback: {motion_fallback_count}, hold: {hold_count}"
            )
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

        # ---------------- PASS 2.6: Dynamic Punch-in Zoom Curve ----------------
        if self.punch_in == "auto":
            z_max = get_max_punch_in_zoom(height)
            if self.config.punch_in_events is not None:
                punch_events = self.config.punch_in_events
            else:
                words_to_use = words if words is not None else self.config.words
                punch_events = detect_punch_in_events(
                    media_path=src_path,
                    words=words_to_use,
                )
            punch_curve = zoom_curve(num_frames, fps, punch_events, z_max=z_max, ramp=0.35)
            total_zoom_arr = zoom * punch_curve
            crop_w_arr = np.clip(base_crop_w / total_zoom_arr, 1.0, float(width)).astype(np.float32)
            crop_h_arr = np.clip(base_crop_h / total_zoom_arr, 1.0, float(height)).astype(np.float32)
            if len(punch_events) > 0:
                print(f"[reframe] Punch-in zoom enabled: {len(punch_events)} event(s), z_max={z_max:.2f}x (height={height}p)")
        else:
            punch_curve = np.ones(num_frames, dtype=np.float32)
            total_zoom_arr = np.full(num_frames, zoom, dtype=np.float32)
            crop_w_arr = np.full(num_frames, float(crop_w), dtype=np.float32)
            crop_h_arr = np.full(num_frames, float(crop_h), dtype=np.float32)

        # ---------------- PASS 2.7: Anti-Corner & Upper-Third Headroom Framing Invariants ----------------
        # 1. Horizontal centering safe-band: presenter's face never strays farther than 18% of crop width from center
        centered_cx = enforce_centering_margin(
            crop_cx=smoothed_cx,
            face_cx=interpolated_cx,
            crop_w=crop_w_arr,
            max_deviation_ratio=0.18
        )
        crop_x0 = np.clip(centered_cx - crop_w_arr / 2.0, 0.0, float(width) - crop_w_arr)

        # 2. Vertical headroom safe-band: keeps eyes in upper third without hitting top or captions
        centered_cy = enforce_headroom_margin(
            crop_cy=smoothed_cy,
            face_cy=interpolated_cy,
            crop_h=crop_h_arr,
            target_headroom=self.headroom,
            max_deviation=0.12
        )
        target_crop_y = centered_cy - self.headroom * crop_h_arr
        crop_y0 = np.clip(target_crop_y, 0.0, float(height) - crop_h_arr)

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
                        round(float(crop_w_arr[i]), 1),
                        round(float(crop_h_arr[i]), 1),
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
            lock_pct=face_lock_pct,
            num_valid_faces=face_lock_count,
            detect_every=detect_every,
            crop_w_arr=crop_w_arr,
            crop_h_arr=crop_h_arr,
            zoom_curve=total_zoom_arr,
            detector=self.detector.name,
            face_lock_count=face_lock_count,
            motion_fallback_count=motion_fallback_count,
            hold_count=hold_count,
        )

    def render(self, plan: Optional[ReframePlan] = None, fast: bool = False) -> Path:
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
        enc_args = ["-c:v", "h264_videotoolbox", "-b:v", "12M"] if fast else ["-c:v", "libx264", "-preset", plan.encoder_preset, "-crf", str(plan.crf)]

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
            ]
            ffmpeg_cmd.extend(enc_args)
            ffmpeg_cmd.extend([
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", plan.audio_bitrate, "-shortest",
                "-movflags", "+faststart", str(out_path)
            ])
        else:
            scale_desc = f"{out_canvas_w}x{out_canvas_h} {'4:5 feed' if is_4_5 else '9:16 vertical'}"
            print(f"[reframe] Encoding {scale_desc} direct-crop video via FFmpeg pipe -> {out_path}")
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_canvas_w}x{out_canvas_h}", "-r", f"{fps}", "-i", "-",
                "-i", str(src_path),
                "-map", "0:v", "-map", "1:a:0",
                "-vf", "format=yuv420p",
                "-af", "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0",
                "-fps_mode", "cfr",
            ]
            ffmpeg_cmd.extend(enc_args)
            ffmpeg_cmd.extend([
                "-c:a", "aac", "-b:a", plan.audio_bitrate, "-shortest",
                "-movflags", "+faststart", str(out_path)
            ])

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
                x0 = float(crop_x0[t]) if t < len(crop_x0) else float(crop_x0[-1])
                y0 = float(crop_y0[t]) if t < len(crop_y0) else float(crop_y0[-1])
                w_t = float(plan.crop_w_arr[t]) if plan.crop_w_arr is not None else float(crop_w)
                h_t = float(plan.crop_h_arr[t]) if plan.crop_h_arr is not None else float(crop_h)

                if plan.layout == "blur-bw":
                    affine_matrix = np.float32([[1, 0, -x0], [0, 1, -y0]])
                    cropped_frame = cv2.warpAffine(frame, affine_matrix, (crop_w, crop_h), flags=cv2.INTER_LINEAR)
                    if proc.stdin:
                        proc.stdin.write(cropped_frame.tobytes())
                else:
                    # Single fused crop + scale warp directly to final canvas size
                    sx = float(out_canvas_w) / w_t
                    sy = float(out_canvas_h) / h_t
                    M = np.float32([
                        [sx, 0, -sx * x0],
                        [0, sy, -sy * y0]
                    ])
                    warped_frame = cv2.warpAffine(frame, M, (out_canvas_w, out_canvas_h), flags=cv2.INTER_CUBIC)
                    if proc.stdin:
                        proc.stdin.write(warped_frame.tobytes())

                if debug_writer is not None:
                    vis = frame.copy()
                    if plan.detections and t < len(plan.detections):
                        for (dx, dy, dw, dh, m) in plan.detections[t]:
                            color = (0, 0, 255) if is_static_detection(dx, dy) else (0, 255, 0)
                            cv2.rectangle(vis, (int(dx - dw / 2), int(dy - dh / 2)),
                                          (int(dx + dw / 2), int(dy + dh / 2)), color, 2)
                    x0_i, y0_i = int(round(x0)), int(round(y0))
                    cw_i, ch_i = int(round(w_t)), int(round(h_t))
                    cv2.rectangle(vis, (x0_i, y0_i), (x0_i + cw_i, y0_i + ch_i), (255, 200, 0), 2)
                    # Upper-third eyeline guideline
                    eyeline_y = int(round(y0_i + self.headroom * ch_i))
                    cv2.line(vis, (x0_i, eyeline_y), (x0_i + cw_i, eyeline_y), (0, 255, 255), 1)
                    # HUD overlay
                    is_cut = t in cut_indices
                    z_val = float(plan.zoom_curve[t]) if plan.zoom_curve is not None else zoom
                    cv2.putText(vis, f"Preset: {self.preset_name} | Zoom: {z_val:.2f}x | Cut: {is_cut}", (20, 40),
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

    def run(self, fast: bool = False) -> Path:
        """Execute reframing: plan passes 1-2.7 and render pass 3 in sequence."""
        plan = self.plan()
        return self.render(plan, fast=fast)


def compute_corner_drift(
    plan_a: ReframePlan,
    plan_b: ReframePlan
) -> dict[str, float]:
    """
    Compute corner drift metrics between two reframing trajectories:
    - max_drift_px: Maximum Euclidean distance of crop top-left corner
    - mean_drift_px: Mean Euclidean distance across all frames
    - max_drift_x: Maximum horizontal corner drift
    - max_drift_y: Maximum vertical corner drift
    """
    n = min(len(plan_a.crop_x0), len(plan_b.crop_x0))
    if n == 0:
        return {"max_drift_px": 0.0, "mean_drift_px": 0.0, "max_drift_x": 0.0, "max_drift_y": 0.0}
    dx = np.abs(plan_a.crop_x0[:n] - plan_b.crop_x0[:n])
    dy = np.abs(plan_a.crop_y0[:n] - plan_b.crop_y0[:n])
    euclid = np.sqrt(dx**2 + dy**2)
    return {
        "max_drift_px": float(np.max(euclid)),
        "mean_drift_px": float(np.mean(euclid)),
        "max_drift_x": float(np.max(dx)),
        "max_drift_y": float(np.max(dy)),
    }


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
    fast: bool = False,
    detect_every: int = 3,
    detect_width: int = 320,
    punch_in: str = "off",
    detector: str = "haar",
    model_path: Optional[Path | str] = None,
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
        audio_delay_ms=audio_delay_ms,
        detect_every=detect_every,
        detect_width=detect_width,
        punch_in=punch_in,
        detector=detector,
        model_path=model_path,
    )
    reframer = FaceReframer(config)
    return reframer.run(fast=fast)


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
    parser.add_argument("--detect-every", type=int, default=3, help="Run face detection every Nth frame with template tracking in between (default: 3)")
    parser.add_argument("--detect-width", type=int, default=320, help="Downscale width for face detection (default: 320)")
    parser.add_argument("--detector", default="haar", choices=["haar", "yunet", "vision"], help="Face detector backend: haar (default), yunet (ONNX neural net), or vision (Apple Vision framework)")
    parser.add_argument("--model-path", default=None, help="Custom path to YuNet ONNX model (face_detection_yunet_2023mar.onnx)")
    parser.add_argument("--debug", action="store_true", help="Generate debug video with bounding box overlays")
    parser.add_argument("--fast", action="store_true", help="Use Apple Silicon hardware acceleration (h264_videotoolbox -b:v 12M)")
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
        audio_delay_ms=args.audio_delay,
        detector=args.detector,
        model_path=args.model_path,
    )
