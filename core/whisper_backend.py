#!/usr/bin/env python3
"""
whisper_backend.py - Centralized Whisper Transcription Engine.

Provides:
1. Cached model loader: loads each Whisper model name once per process in memory.
2. Unified transcribe_words() returning Word(start, end, text) and segment text.
3. Dual-backend support:
   - "openai": Standard openai-whisper (CPU, fp16=False on Apple Silicon).
   - "mlx": Apple Silicon GPU accelerated mlx-whisper (optional, used if available).
   - "auto": Prefers mlx if installed, otherwise gracefully falls back to openai.
4. Preserves all decoding invariants:
   - language="hi"
   - no_speech_threshold=0.3
   - condition_on_previous_text=False
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Word:
    """Word-level alignment timestamp."""
    start: float
    end: float
    text: str

    @property
    def word(self) -> str:
        """Alias for backward-compatibility with dictionary/attribute access."""
        return self.text


class TranscriptionResult(tuple):
    """
    Unified result object supporting both:
    1. Tuple unpacking: words, segment_text = transcribe_words(...)
    2. Property access: res.words, res.segments, res.text, res.segment_text
    """
    words: List[Word]
    segments: List[Dict[str, Any]]
    text: str
    segment_text: str

    def __new__(cls, words: List[Word], segments: List[Dict[str, Any]], text: str):
        instance = super().__new__(cls, (words, text))
        instance.words = words
        instance.segments = segments
        instance.text = text
        instance.segment_text = text
        return instance


# ── Backend Availability & HF Model Repositories ─────────────────────────────

HAS_MLX_WHISPER: bool = False
try:
    import mlx.core as _mx
    import mlx_whisper as _mlx_whisper
    HAS_MLX_WHISPER = True
except (ImportError, Exception):
    HAS_MLX_WHISPER = False


# Map standard Whisper shorthand names to MLX HuggingFace repository IDs
MLX_MODEL_MAP: Dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

# ── In-Memory Model Cache ─────────────────────────────────────────────────────
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_LOAD_COUNTS: Dict[str, int] = {}


def resolve_backend(backend: str = "auto") -> str:
    """Normalize and resolve requested backend against system capabilities."""
    b = (backend or "auto").strip().lower()
    if b in ("auto", "default"):
        return "mlx" if HAS_MLX_WHISPER else "openai"
    if b in ("mlx", "mlx_whisper", "mlx-whisper"):
        if not HAS_MLX_WHISPER:
            print("[whisper_backend] Warning: MLX requested but not available. Falling back to openai.")
            return "openai"
        return "mlx"
    return "openai"


def get_whisper_model(model_name: str = "large-v3", backend: str = "auto") -> Any:
    """
    Cached model loader: loads each (model_name, backend) exactly once per process.
    Subsequent calls return the existing in-memory instance instantly.
    """
    resolved_backend = resolve_backend(backend)
    cache_key = (model_name, resolved_backend)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if resolved_backend == "openai":
        import whisper
        print(f"[whisper_backend] Loading OpenAI Whisper model '{model_name}' (device=cpu, fp16=False)...")
        loaded_model = whisper.load_model(model_name)
    elif resolved_backend == "mlx":
        import mlx_whisper
        repo_name = MLX_MODEL_MAP.get(model_name, f"mlx-community/whisper-{model_name}")
        print(f"[whisper_backend] Loading MLX Whisper model '{model_name}' ({repo_name} on GPU)...")
        # Load weights once to warm cache
        loaded_model = mlx_whisper.load_models.load_model(repo_name)
    else:
        raise ValueError(f"Unknown whisper backend: {resolved_backend}")

    _MODEL_CACHE[cache_key] = loaded_model
    _MODEL_LOAD_COUNTS[model_name] = _MODEL_LOAD_COUNTS.get(model_name, 0) + 1
    backend_key = f"{model_name}:{resolved_backend}"
    _MODEL_LOAD_COUNTS[backend_key] = _MODEL_LOAD_COUNTS.get(backend_key, 0) + 1
    return loaded_model


def get_model_load_count(model_name: str, backend: Optional[str] = None) -> int:
    """Return how many times a given model was freshly loaded from disk."""
    if backend is not None:
        resolved_backend = resolve_backend(backend)
        key = f"{model_name}:{resolved_backend}"
        return _MODEL_LOAD_COUNTS.get(key, _MODEL_LOAD_COUNTS.get(model_name, 0))
    return _MODEL_LOAD_COUNTS.get(model_name, 0)


# ── Unified Word-Level Transcription ─────────────────────────────────────────

def transcribe_words(
    audio_path: Path | str,
    language: str = "hi",
    model: str = "large-v3",
    initial_prompt: Optional[str] = None,
    backend: str = "auto",
    word_timestamps: bool = True,
    no_speech_threshold: float = 0.3,
    condition_on_previous_text: bool = False,
    fp16: bool = False,
) -> TranscriptionResult:
    """
    Transcribe audio/video file with word-level timestamps and segment text.

    Args:
        audio_path: Path to video (mp4, mov) or audio (mp3, wav) file.
        language: Language code (default 'hi').
        model: Whisper model size name ('base', 'small', 'large-v3', etc.).
        initial_prompt: Optional transcription guidance prompt.
        backend: 'auto', 'openai', or 'mlx'.
        word_timestamps: Whether to extract word-level timestamps.
        no_speech_threshold: Silence probability threshold (default 0.3).
        condition_on_previous_text: Disables failure loops (default False).
        fp16: Floating point precision flag (default False for CPU stability).

    Returns:
        TranscriptionResult object containing .words, .segments, and .text.
    """
    path_obj = Path(audio_path).expanduser().resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio/video file not found: {path_obj}")

    resolved_backend = resolve_backend(backend)

    if resolved_backend == "mlx":
        import mlx_whisper
        repo_name = MLX_MODEL_MAP.get(model, f"mlx-community/whisper-{model}")

        # Warm up cached loader
        get_whisper_model(model_name=model, backend="mlx")

        mlx_kwargs: Dict[str, Any] = {
            "path_or_hf_repo": repo_name,
            "language": language,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
            "fp16": fp16,
        }
        if initial_prompt:
            mlx_kwargs["initial_prompt"] = initial_prompt

        raw_result = mlx_whisper.transcribe(str(path_obj), **mlx_kwargs)

        # Retry once without initial_prompt if prompt caused 0 segments
        if initial_prompt and not raw_result.get("segments"):
            mlx_kwargs.pop("initial_prompt", None)
            raw_result = mlx_whisper.transcribe(str(path_obj), **mlx_kwargs)

    else:
        # OpenAI Whisper backend
        wm = get_whisper_model(model_name=model, backend="openai")
        op_kwargs: Dict[str, Any] = {
            "language": language,
            "task": "transcribe",
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
            "fp16": fp16,
        }
        if initial_prompt:
            op_kwargs["initial_prompt"] = initial_prompt

        raw_result = wm.transcribe(str(path_obj), **op_kwargs)

        # Retry once without initial_prompt if prompt caused 0 segments
        if initial_prompt and not raw_result.get("segments"):
            op_kwargs.pop("initial_prompt", None)
            raw_result = wm.transcribe(str(path_obj), **op_kwargs)

    # Normalize words and segments across backends
    words: List[Word] = []
    segments: List[Dict[str, Any]] = []

    for idx, seg in enumerate(raw_result.get("segments", [])):
        seg_text = seg.get("text", "").strip()
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))

        segments.append({
            "id": seg.get("id", idx),
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg_text,
        })

        if word_timestamps:
            for w in seg.get("words", []):
                cleaned_word = w.get("word", "").strip()
                if cleaned_word:
                    words.append(Word(
                        start=round(float(w.get("start", 0.0)), 3),
                        end=round(float(w.get("end", 0.0)), 3),
                        text=cleaned_word,
                    ))

    full_text = raw_result.get("text", "").strip()
    return TranscriptionResult(words=words, segments=segments, text=full_text)
