#!/usr/bin/env python3
"""
download_models.py - Download and verify vision & neural models for PW pipeline.

Downloads:
- face_detection_yunet_2023mar.onnx (OpenCV Zoo YuNet face detector, ~232 KB)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with user-agent and progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    print(f"[download] Fetching {dest_path.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

    try:
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as f:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 16384
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = 100.0 * downloaded / total_size
                    print(f"\r[download] Progress: {downloaded}/{total_size} bytes ({pct:.1f}%)", end="", flush=True)
        print()
        tmp_path.replace(dest_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return dest_path


def ensure_yunet_model(target_dir: Optional[Path | str] = None) -> Path:
    """
    Ensure YuNet ONNX model is available locally.
    Searches standard paths; if missing, downloads from official OpenCV Zoo.
    """
    dest_dir = Path(target_dir).resolve() if target_dir else DEFAULT_MODELS_DIR
    model_path = dest_dir / YUNET_FILENAME

    if model_path.is_file() and model_path.stat().st_size > 100000:
        return model_path

    # Check alternative common locations
    alt_paths = [
        Path.cwd() / "models" / YUNET_FILENAME,
        Path.cwd() / "core" / YUNET_FILENAME,
        Path.home() / ".cache" / "models" / YUNET_FILENAME,
    ]
    for alt in alt_paths:
        if alt.is_file() and alt.stat().st_size > 100000:
            return alt

    # Download to dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    download_file(YUNET_URL, model_path)

    # Verification: check with cv2.FaceDetectorYN
    try:
        import cv2
        detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320))
        if detector is None:
            raise RuntimeError("cv2.FaceDetectorYN.create returned None")
        print(f"[download] Successfully verified YuNet model at: {model_path}")
    except Exception as exc:
        print(f"[download] WARNING: Could not verify YuNet model with OpenCV: {exc}")

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pre-trained vision models for PW pipeline.")
    parser.add_argument(
        "--model",
        default="yunet",
        choices=["yunet", "all"],
        help="Model to download (default: yunet)"
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_MODELS_DIR),
        help=f"Destination directory (default: {DEFAULT_MODELS_DIR})"
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    if args.model in ("yunet", "all"):
        model_path = ensure_yunet_model(dest_dir)
        print(f"[download] Ready: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
