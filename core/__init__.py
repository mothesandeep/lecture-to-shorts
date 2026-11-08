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
    ReframePlan,
    compute_corner_drift,
    find_cascade,
    list_reframe_presets,
    reframe_video,
)
from core.render import render_reel
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
    compute_lecture_cache_key,
    get_lecture_cache_path,
    get_whisper_model,
    load_lecture_transcript_cache,
    map_timestamp_to_clip,
    map_word_to_clip,
    save_lecture_transcript_cache,
    slice_cached_words_for_clip,
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
    "ReframePlan",
    "compute_corner_drift",
    "render_reel",
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
    "compute_lecture_cache_key",
    "get_lecture_cache_path",
    "save_lecture_transcript_cache",
    "load_lecture_transcript_cache",
    "map_timestamp_to_clip",
    "map_word_to_clip",
    "slice_cached_words_for_clip",
]

