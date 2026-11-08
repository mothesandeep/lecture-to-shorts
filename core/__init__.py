"""
core - Internal modules for the lecture-to-reel processing pipeline.
"""

from core.captioner import (
    CAPTION_STYLES,
    CaptionChunk,
    CaptionRenderer,
    CaptionStyle,
    FontManager,
    burn_captions,
    get_style,
    list_styles,
)
from core.av_sync import (
    cut_snippet_accurate,
    ensure_cfr_video,
    probe_av_sync,
    verify_av_sync,
)
from core.health_check import run_diagnostics
from core.hook_finder import (
    HookFinder,
    ViralClipMetadata,
    cut_snippet_without_reencode,
    extract_audio,
    format_seconds_to_timestamp,
)
from core.reframer import (
    REFRAME_PRESETS,
    FaceReframer,
    ReframeConfig,
    find_cascade,
    list_reframe_presets,
    reframe_video,
)
from core.transcriber import (
    CACHE_DIR,
    DEV,
    HinglishTranscriber,
    WordCaption,
    local_romanize,
    sanitize_latin,
)
from core.whisper_backend import (
    HAS_MLX_WHISPER,
    TranscriptionResult,
    Word,
    get_whisper_model,
    transcribe_words,
)

__all__ = [
    "CAPTION_STYLES",
    "CaptionChunk",
    "CaptionRenderer",
    "CaptionStyle",
    "FontManager",
    "burn_captions",
    "get_style",
    "list_styles",
    "run_diagnostics",
    "HookFinder",
    "ViralClipMetadata",
    "cut_snippet_without_reencode",
    "cut_snippet_accurate",
    "ensure_cfr_video",
    "probe_av_sync",
    "verify_av_sync",
    "extract_audio",
    "format_seconds_to_timestamp",
    "FaceReframer",
    "REFRAME_PRESETS",
    "ReframeConfig",
    "find_cascade",
    "list_reframe_presets",
    "reframe_video",
    "CACHE_DIR",
    "DEV",
    "HinglishTranscriber",
    "WordCaption",
    "local_romanize",
    "sanitize_latin",
    "Word",
    "TranscriptionResult",
    "transcribe_words",
    "get_whisper_model",
    "HAS_MLX_WHISPER",
]

