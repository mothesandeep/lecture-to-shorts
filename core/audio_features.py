#!/usr/bin/env python3
"""
audio_features.py - Audio feature extraction for transcript annotation and viral hook ranking.

Extracts:
1. Per-window RMS energy z-scores (5-second windows) to identify high-energy, passionate moments.
2. Speech rate (words/second) from word-level Whisper timestamps (fast vs slow pacing).
3. Pause density (ratio of speech gaps >= 0.35s to total duration) to detect punchy vs hesitant speech.

Uses only FFmpeg and NumPy (standard library + NumPy).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_audio_samples(
    audio_or_video_path: Path | str,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extract raw 16kHz mono 16-bit PCM audio samples from an audio or video file
    using an FFmpeg pipe. Returns float32 array in [-1.0, 1.0].
    """
    p = Path(audio_or_video_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio/video file not found: {p}")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-vn",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_data, stderr_data = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg PCM audio extraction failed: {stderr_data.decode('utf-8', errors='ignore')}")

    if not raw_data:
        return np.array([], dtype=np.float32)

    int_samples = np.frombuffer(raw_data, dtype=np.int16)
    return int_samples.astype(np.float32) / 32768.0


class AudioFeatureExtractor:
    """
    Computes acoustic and prosodic features from lecture audio and word-level timestamps:
    - RMS energy z-scores across 5s sliding windows.
    - Local speech rate (words/second).
    - Pause density (inter-word silence ratio).
    """

    def __init__(
        self,
        samples: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        window_duration_s: float = 5.0,
        hop_duration_s: float = 2.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_duration_s = window_duration_s
        self.hop_duration_s = hop_duration_s

        self.samples = samples if samples is not None else np.array([], dtype=np.float32)
        self.window_centers: np.ndarray = np.array([])
        self.energy_zscores: np.ndarray = np.array([])
        self.mean_rms: float = 0.0
        self.std_rms: float = 1.0

        if len(self.samples) > 0:
            self._compute_energy_profile()

    @classmethod
    def from_file(
        cls,
        audio_or_video_path: Path | str,
        sample_rate: int = 16000,
        window_duration_s: float = 5.0,
        hop_duration_s: float = 2.5,
    ) -> "AudioFeatureExtractor":
        """Instantiate extractor by reading samples from a media file."""
        samples = extract_audio_samples(audio_or_video_path, sample_rate=sample_rate)
        return cls(
            samples=samples,
            sample_rate=sample_rate,
            window_duration_s=window_duration_s,
            hop_duration_s=hop_duration_s
        )

    def _compute_energy_profile(self) -> None:
        """Calculate RMS energy for sliding 5-second windows and compute z-scores."""
        win_size = int(round(self.window_duration_s * self.sample_rate))
        hop_size = int(round(self.hop_duration_s * self.sample_rate))

        total_samples = len(self.samples)
        if total_samples < win_size:
            # Short audio: single window
            rms = float(np.sqrt(np.mean(self.samples**2) + 1e-12))
            self.window_centers = np.array([total_samples / (2.0 * self.sample_rate)])
            self.energy_zscores = np.array([0.0])
            self.mean_rms = rms
            self.std_rms = 1.0
            return

        centers = []
        rms_values = []
        for start in range(0, total_samples - win_size + 1, hop_size):
            chunk = self.samples[start:start + win_size]
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            center_s = (start + win_size / 2.0) / self.sample_rate
            centers.append(center_s)
            rms_values.append(rms)

        self.window_centers = np.array(centers, dtype=np.float32)
        rms_arr = np.array(rms_values, dtype=np.float32)

        self.mean_rms = float(np.mean(rms_arr))
        self.std_rms = float(np.std(rms_arr))
        if self.std_rms < 1e-6:
            self.std_rms = 1.0

        self.energy_zscores = (rms_arr - self.mean_rms) / self.std_rms

    def get_energy_zscore(self, start_t: float, end_t: float) -> float:
        """
        Get the average RMS energy z-score covering the given time range [start_t, end_t].
        """
        if len(self.window_centers) == 0:
            return 0.0

        # Windows with centers inside the segment
        mask_inside = (self.window_centers >= start_t) & (self.window_centers <= end_t)
        if np.any(mask_inside):
            return float(np.mean(self.energy_zscores[mask_inside]))

        # Windows overlapping with segment bounds
        mask_overlap = (self.window_centers >= start_t - self.hop_duration_s) & (self.window_centers <= end_t + self.hop_duration_s)
        if np.any(mask_overlap):
            return float(np.mean(self.energy_zscores[mask_overlap]))

        # Nearest neighbor fallback to segment midpoint
        mid_t = (start_t + end_t) / 2.0
        idx = int(np.argmin(np.abs(self.window_centers - mid_t)))
        return float(self.energy_zscores[idx])

    @staticmethod
    def compute_speech_rate(words: List[Any], start_t: float, end_t: float) -> float:
        """
        Compute speech rate (words per second) within [start_t, end_t].
        Supports Word objects (start, end, text) or dicts.
        """
        duration = max(0.2, end_t - start_t)
        count = 0
        for w in words:
            w_start = getattr(w, "start", None) if hasattr(w, "start") else w.get("start")
            w_end = getattr(w, "end", None) if hasattr(w, "end") else w.get("end")
            if w_start is None or w_end is None:
                continue
            if w_end >= start_t and w_start <= end_t:
                count += 1
        return count / duration

    @staticmethod
    def compute_pause_density(
        words: List[Any],
        start_t: float,
        end_t: float,
        pause_threshold_s: float = 0.35
    ) -> float:
        """
        Compute pause density (fraction of duration spent in pauses >= pause_threshold_s)
        between words within [start_t, end_t].
        """
        duration = max(0.2, end_t - start_t)
        seg_words = []
        for w in words:
            w_start = getattr(w, "start", None) if hasattr(w, "start") else w.get("start")
            w_end = getattr(w, "end", None) if hasattr(w, "end") else w.get("end")
            if w_start is None or w_end is None:
                continue
            if w_end >= start_t and w_start <= end_t:
                seg_words.append((max(start_t, float(w_start)), min(end_t, float(w_end))))

        if not seg_words:
            return 1.0  # Entire segment is silent/paused

        seg_words.sort(key=lambda x: x[0])
        pause_duration = 0.0

        # Leading pause
        leading = seg_words[0][0] - start_t
        if leading >= pause_threshold_s:
            pause_duration += leading

        # Inter-word pauses
        for i in range(len(seg_words) - 1):
            gap = seg_words[i + 1][0] - seg_words[i][1]
            if gap >= pause_threshold_s:
                pause_duration += gap

        # Trailing pause
        trailing = end_t - seg_words[-1][1]
        if trailing >= pause_threshold_s:
            pause_duration += trailing

        return min(1.0, pause_duration / duration)

    def annotate_tags(
        self,
        start_t: float,
        end_t: float,
        words: Optional[List[Any]] = None
    ) -> List[str]:
        """
        Generate feature tags for a segment, e.g. ['energy +2.1σ', 'fast', 'punchy'].
        """
        tags: List[str] = []

        # 1. Energy z-score
        z = self.get_energy_zscore(start_t, end_t)
        if z >= 0.0:
            tags.append(f"energy +{z:.1f}σ")
        else:
            tags.append(f"energy {z:.1f}σ")

        if words:
            # 2. Speech rate (words/sec)
            rate = self.compute_speech_rate(words, start_t, end_t)
            if rate >= 3.6:
                tags.append("fast")
            elif rate <= 1.8:
                tags.append("slow")

            # 3. Pause density
            pause_density = self.compute_pause_density(words, start_t, end_t)
            if pause_density <= 0.12:
                tags.append("punchy")
            elif pause_density >= 0.35:
                tags.append("pauses")

        return tags

    def format_annotated_segment(
        self,
        seg_id_str: str,
        start_t: float,
        end_t: float,
        text: str,
        words: Optional[List[Any]] = None
    ) -> str:
        """
        Format a single transcript line with sequential ID, time, acoustic/pacing tags, and text:
        [S0001] [00:00 - 00:04 | 0.0s - 4.2s] [energy +2.1σ, fast] Some spoken words...
        """
        mins_s, secs_s = int(start_t // 60), int(start_t % 60)
        mins_e, secs_e = int(end_t // 60), int(end_t % 60)
        time_tag = f"[{mins_s:02d}:{secs_s:02d} - {mins_e:02d}:{secs_e:02d} | {start_t:.1f}s - {end_t:.1f}s]"

        tags = self.annotate_tags(start_t, end_t, words=words)
        tag_str = f"[{', '.join(tags)}]" if tags else ""

        parts = [seg_id_str, time_tag]
        if tag_str:
            parts.append(tag_str)
        parts.append(text.strip())

        return " ".join(parts)
