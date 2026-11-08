#!/usr/bin/env python3
"""
captioner.py - Pillow-to-FFmpeg word-by-word caption burner.

Burns animated word-by-word captions (yellow active word, white inactive words,
black stroke outline) directly into video frames piped to FFmpeg rawvideo stdin.

Features integrated from Caption-AI (https://github.com/morningstarwebd/caption-ai):
- Multiple visual presets (classic-yellow, box-highlight, bold-stroke, fade-in,
  kalakar-fire, karaoke-neon, cyber-neon, cinematic, comic-pop, minimal).
- Word-bounce / pop scaling animations on active word.
- Bounding box / rounded pill background highlights (active-word pill and chunk container).
- Multi-layer 3D drop-shadows and thick stroke outlines.
- Opacity hierarchy (dimmed upcoming words) and smooth chunk entry fades.
- Font family hierarchy (sans, impact, serif, mono).

Key Invariants:
1. Zero external subtitle engine dependency (No libass, no drawtext filter).
2. Dynamic font size auto-fitting within video safe boundaries (W - 80).
3. Active word highlighted in bright yellow (255, 255, 0) with 4px black stroke (default style).
4. Strict Latin-only invariant asserting no Devanagari reaches the renderer.
5. Exact audio preservation without drift (copies audio stream from input video).
"""

from __future__ import annotations

import copy
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from core.transcriber import DEV, WordCaption
    from core.av_sync import ensure_cfr_video, probe_av_sync, verify_av_sync
except ImportError:
    from transcriber import DEV, WordCaption
    from av_sync import ensure_cfr_video, probe_av_sync, verify_av_sync


@dataclass
class ChunkLayoutMeta:
    """Computed layout metadata for a group of words in a caption chunk."""
    font_size: int
    space_width: float
    word_widths: List[float]
    total_width: float


@dataclass
class CaptionChunk:
    """Group of words displayed together on screen."""
    words: List[WordCaption]
    meta: Optional[ChunkLayoutMeta] = None

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0


@dataclass
class CaptionSprite:
    """Precomputed RGBA sprite for a single (chunk, active_word_idx) state."""
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    sprite_pre: Optional[np.ndarray] = None    # Premultiplied float32 BGR: (H, W, 3)
    inv_alpha: Optional[np.ndarray] = None     # Inverse float32 alpha: (H, W, 1)
    alpha_f: Optional[np.ndarray] = None       # Alpha [0, 1] float32: (H, W, 1)
    sprite_bgr_f: Optional[np.ndarray] = None  # Non-premultiplied float32 BGR: (H, W, 3)


@dataclass
class CaptionStyle:
    """
    Visual style configuration for caption rendering.
    Inspired by Caption-AI themes (viral_shorts, kalakar_fire, karaoke_neon, highlight_box, etc.).
    """
    name: str
    description: str = ""
    active_color: Tuple[int, int, int] = (255, 255, 0)         # Bright Yellow RGB
    inactive_color: Tuple[int, int, int] = (255, 255, 255)     # Crisp White RGB
    stroke_color: Tuple[int, int, int] = (0, 0, 0)             # Solid Black RGB
    stroke_width: int = 4
    font_family: str = "sans"                                  # "sans", "impact", "serif", "mono"
    uppercase: bool = True

    # Word bounce / scale pop effect (Caption-AI word-pop: scale(1.15) translateY(-1px))
    word_bounce: bool = False
    bounce_scale: float = 1.12
    bounce_y_offset: int = -3

    # Active word pill/box highlight (Caption-AI highlight_box)
    active_box_highlight: bool = False
    active_box_fill: Tuple[int, int, int, int] = (250, 204, 21, 255)  # Vibrant Yellow RGBA (#FACC15)
    active_box_text_color: Tuple[int, int, int] = (0, 0, 0)           # Text inside active pill
    active_box_radius: int = 8
    active_box_padding: Tuple[int, int] = (12, 6)                     # (pad_x, pad_y)

    # Chunk container backdrop pill (Caption-AI minimal / glass / karaoke)
    chunk_bg: bool = False
    chunk_bg_fill: Tuple[int, int, int, int] = (0, 0, 0, 150)         # RGBA
    chunk_bg_border: Optional[Tuple[int, int, int, int]] = None
    chunk_bg_radius: int = 14
    chunk_bg_padding: Tuple[int, int] = (20, 10)

    # Drop shadow / 3D depth (Caption-AI shadow_3d / comic_pop)
    shadow_offset: Optional[Tuple[int, int]] = None                    # (dx, dy)
    shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 220)

    # 3D Extrusion Shadow (AE beveled/layered physical depth)
    shadow_3d: bool = False
    shadow_3d_color: Tuple[int, int, int] = (4, 32, 85)                # Deep Navy Blue
    shadow_3d_depth: int = 4

    # Glow / Optical Bloom (AE Deep Glow / Stylize Glow simulation)
    glow: bool = False
    glow_color: Tuple[int, int, int] = (0, 220, 255)                   # Vibrant Electric Cyan
    glow_radius: int = 14
    glow_stroke: int = 8

    # Opacity hierarchy & Fade (Caption-AI fade-in / karaoke)
    future_alpha: int = 255                                           # Alpha for upcoming words
    past_alpha: int = 255                                             # Alpha for already spoken words
    chunk_fade_in: bool = False                                       # Alpha ramp on chunk entry
    chunk_fade_duration: float = 0.12                                 # Seconds
    bottom_offset_adjust: int = 0                                     # Pixel shift relative to bottom_offset


# ── Preset Style Catalog ─────────────────────────────────────────────────────────────

CAPTION_STYLES: Dict[str, CaptionStyle] = {
    "classic-yellow": CaptionStyle(
        name="classic-yellow",
        description="Proven viral active-yellow highlight with white words and black stroke (Default).",
        active_color=(255, 255, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=4,
        font_family="sans",
        uppercase=True,
        word_bounce=False,
    ),
    "box-highlight": CaptionStyle(
        name="box-highlight",
        description="Trending viral reel style with bright yellow rounded box behind the active word.",
        active_color=(0, 0, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=4,
        font_family="sans",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.08,
        bounce_y_offset=-2,
        active_box_highlight=True,
        active_box_fill=(250, 204, 21, 255),
        active_box_text_color=(0, 0, 0),
        active_box_radius=8,
        active_box_padding=(12, 6),
    ),
    "bold-stroke": CaptionStyle(
        name="bold-stroke",
        description="Heavy 7px stroke with 3D drop-shadow for intense contrast on complex backgrounds.",
        active_color=(255, 220, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=7,
        font_family="sans",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.12,
        bounce_y_offset=-4,
        shadow_offset=(4, 4),
        shadow_color=(0, 0, 0, 255),
    ),
    "fade-in": CaptionStyle(
        name="fade-in",
        description="Smooth opacity hierarchy with dimmed upcoming words and active word pop.",
        active_color=(255, 255, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=4,
        font_family="sans",
        uppercase=True,
        future_alpha=90,
        past_alpha=255,
        chunk_fade_in=True,
        chunk_fade_duration=0.12,
        word_bounce=True,
        bounce_scale=1.10,
        bounce_y_offset=-2,
    ),
    "kalakar-fire": CaptionStyle(
        name="kalakar-fire",
        description="Caption-AI inspired fiery orange highlight with warm glow shadow.",
        active_color=(255, 107, 53),
        inactive_color=(255, 245, 235),
        stroke_color=(20, 5, 5),
        stroke_width=5,
        font_family="sans",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.12,
        bounce_y_offset=-4,
        shadow_offset=(3, 3),
        shadow_color=(180, 40, 10, 200),
    ),
    "karaoke-neon": CaptionStyle(
        name="karaoke-neon",
        description="Electric neon green active word inside a rounded dark translucent backdrop pill.",
        active_color=(0, 255, 136),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=3,
        font_family="sans",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.10,
        bounce_y_offset=-2,
        chunk_bg=True,
        chunk_bg_fill=(0, 0, 0, 160),
        chunk_bg_border=(0, 255, 136, 70),
        chunk_bg_radius=14,
        chunk_bg_padding=(20, 10),
    ),
    "cyber-neon": CaptionStyle(
        name="cyber-neon",
        description="Electric cyan active word with vibrant purple/magenta shadow halo.",
        active_color=(0, 255, 255),
        inactive_color=(245, 245, 255),
        stroke_color=(0, 0, 0),
        stroke_width=4,
        font_family="sans",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.12,
        bounce_y_offset=-3,
        shadow_offset=(2, 2),
        shadow_color=(200, 0, 200, 200),
    ),
    "cinematic": CaptionStyle(
        name="cinematic",
        description="Serif typography (Georgia/Times), champagne gold highlight, dark letterbox pill.",
        active_color=(240, 230, 140),
        inactive_color=(220, 215, 205),
        stroke_color=(0, 0, 0),
        stroke_width=2,
        font_family="serif",
        uppercase=True,
        word_bounce=False,
        chunk_bg=True,
        chunk_bg_fill=(15, 15, 20, 190),
        chunk_bg_radius=6,
        chunk_bg_padding=(24, 12),
        bottom_offset_adjust=-30,
    ),
    "comic-pop": CaptionStyle(
        name="comic-pop",
        description="Impact bold font, vibrant yellow text with 3D black drop shadow.",
        active_color=(255, 240, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=4,
        font_family="impact",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.15,
        bounce_y_offset=-4,
        shadow_offset=(5, 5),
        shadow_color=(0, 0, 0, 255),
    ),
    "minimal": CaptionStyle(
        name="minimal",
        description="Clean translucent dark pill with crisp typography and soft gold active highlight.",
        active_color=(255, 230, 90),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=1,
        font_family="sans",
        uppercase=True,
        word_bounce=False,
        chunk_bg=True,
        chunk_bg_fill=(0, 0, 0, 150),
        chunk_bg_radius=12,
        chunk_bg_padding=(18, 8),
    ),
    "subtitle-16x9": CaptionStyle(
        name="subtitle-16x9",
        description="Clean 16:9 widescreen bottom subtitle with dark translucent pill and crisp active word highlight.",
        active_color=(255, 240, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=2,
        font_family="sans",
        uppercase=False,
        word_bounce=False,
        chunk_bg=True,
        chunk_bg_fill=(0, 0, 0, 160),
        chunk_bg_radius=8,
        chunk_bg_padding=(20, 8),
        bottom_offset_adjust=0,
    ),
    "ae-glow": CaptionStyle(
        name="ae-glow",
        description="After Effects aesthetic: Montserrat font, 3D deep navy shadow, and electric cyan neon glow.",
        active_color=(0, 220, 255),               # Electric Cyan RGB (#00DCFF)
        inactive_color=(255, 255, 255),           # Pure Crisp White
        stroke_color=(4, 32, 85),                 # Deep Navy Blue Stroke
        stroke_width=2,
        font_family="montserrat",
        uppercase=True,
        word_bounce=True,
        bounce_scale=1.10,
        bounce_y_offset=-3,
        shadow_offset=(2, 2),
        shadow_color=(0, 0, 0, 180),
        shadow_3d=True,
        shadow_3d_color=(4, 32, 85),
        shadow_3d_depth=4,
        glow=True,
        glow_color=(0, 220, 255),
        glow_radius=14,
        glow_stroke=8,
    ),
}

STYLE_ALIASES: Dict[str, str] = {
    "ae": "ae-glow",
    "ae-glow": "ae-glow",
    "ae-cyan": "ae-glow",
    "after-effects": "ae-glow",
    "cyan-glow": "ae-glow",
    "twisted": "ae-glow",
    "classic": "classic-yellow",
    "yellow": "classic-yellow",
    "active-yellow": "classic-yellow",
    "default": "classic-yellow",
    "box": "box-highlight",
    "highlight-box": "box-highlight",
    "pill": "box-highlight",
    "bold": "bold-stroke",
    "heavy": "bold-stroke",
    "outline-bold": "bold-stroke",
    "fade": "fade-in",
    "fadein": "fade-in",
    "fire": "kalakar-fire",
    "kalakar": "kalakar-fire",
    "neon": "karaoke-neon",
    "green-neon": "karaoke-neon",
    "neon-green": "karaoke-neon",
    "cyber": "cyber-neon",
    "neon-glow": "cyber-neon",
    "cinema": "cinematic",
    "comic": "comic-pop",
    "widescreen": "subtitle-16x9",
    "horizontal": "subtitle-16x9",
    "subtitles": "subtitle-16x9",
    "16x9": "subtitle-16x9",
    "16:9": "subtitle-16x9",
}


def list_styles() -> List[str]:
    """Return list of all available style preset names."""
    return list(CAPTION_STYLES.keys())


def get_style(style_input: Optional[Union[str, CaptionStyle]] = None) -> CaptionStyle:
    """Resolve string name or alias into a deep copy of a CaptionStyle instance."""
    if isinstance(style_input, CaptionStyle):
        return copy.deepcopy(style_input)
    if not style_input:
        return copy.deepcopy(CAPTION_STYLES["classic-yellow"])

    key = str(style_input).strip().lower().replace("_", "-")
    resolved = STYLE_ALIASES.get(key, key)
    if resolved in CAPTION_STYLES:
        return copy.deepcopy(CAPTION_STYLES[resolved])

    available = ", ".join(CAPTION_STYLES.keys())
    raise ValueError(f"Unknown caption style '{style_input}'. Available presets: {available}")


class FontManager:
    """Caches and resolves TrueType fonts across system families and custom paths."""

    FAMILY_CANDIDATES: Dict[str, List[str]] = {
        "montserrat": [
            "/Library/Fonts/Montserrat-ExtraBold.ttf",
            "/Library/Fonts/Montserrat-Black.ttf",
            "/Library/Fonts/Montserrat-Bold.ttf",
            "/Library/Fonts/Montserrat-SemiBold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        "sans": [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ],
        "impact": [
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        "serif": [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        ],
        "mono": [
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ],
    }

    DEFAULT_FONT_CANDIDATES = FAMILY_CANDIDATES["sans"]

    def __init__(self, custom_font_path: Optional[str | Path] = None) -> None:
        self.custom_font_path = Path(custom_font_path) if custom_font_path else None
        self._cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get_font(self, size: int, font_family: Optional[str] = None) -> ImageFont.FreeTypeFont:
        """Retrieve cached font instance at specified point size and font family."""
        family_key = (font_family or "sans").lower()
        cache_key = (family_key, size)
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidates: List[Optional[Path]] = []
        if self.custom_font_path:
            candidates.append(self.custom_font_path.expanduser().resolve())

        family_paths = self.FAMILY_CANDIDATES.get(family_key, self.DEFAULT_FONT_CANDIDATES)
        candidates.extend([Path(p) for p in family_paths])
        candidates.extend([Path(p) for p in self.DEFAULT_FONT_CANDIDATES])

        for p in candidates:
            if p and p.is_file():
                try:
                    loaded = ImageFont.truetype(str(p), size)
                    self._cache[cache_key] = loaded
                    return loaded
                except Exception:
                    continue

        # Fallback to default PIL bitmap font
        try:
            loaded = ImageFont.load_default(size=size)  # type: ignore
        except TypeError:
            loaded = ImageFont.load_default()
        self._cache[cache_key] = loaded
        return loaded


class CaptionRenderer:
    """Burns word-by-word animated captions onto vertical video frames."""

    def __init__(
        self,
        font_path: Optional[str | Path] = None,
        chunk_size: int = 4,
        bottom_offset: int = 380,
        active_color: Tuple[int, int, int] = (255, 255, 0),    # Bright Yellow
        inactive_color: Tuple[int, int, int] = (255, 255, 255), # Crisp White
        stroke_color: Tuple[int, int, int] = (0, 0, 0),         # Solid Black
        stroke_width: int = 4,
        preset: str = "fast",
        crf: int = 18,
        encoder: str = "libx264",
        title_banner: Optional[str] = None,
        style: Optional[Union[str, CaptionStyle]] = None
    ) -> None:
        self.font_mgr = FontManager(font_path)
        self.chunk_size = chunk_size
        self.bottom_offset = bottom_offset
        self.preset = preset
        self.crf = crf
        self.encoder = encoder
        self.title_banner = title_banner.strip().upper() if title_banner else None

        # Resolve style configuration
        base_style = get_style(style or "classic-yellow")
        if active_color != (255, 255, 0) and style is None:
            base_style.active_color = active_color
        if inactive_color != (255, 255, 255) and style is None:
            base_style.inactive_color = inactive_color
        if stroke_color != (0, 0, 0) and style is None:
            base_style.stroke_color = stroke_color
        if stroke_width != 4 and style is None:
            base_style.stroke_width = stroke_width

        self.style = base_style
        self.active_color = self.style.active_color
        self.inactive_color = self.style.inactive_color
        self.stroke_color = self.style.stroke_color
        self.stroke_width = self.style.stroke_width

    def build_chunks(self, words: List[WordCaption]) -> List[CaptionChunk]:
        """Group individual words into presentation chunks of chunk_size."""
        chunks: List[CaptionChunk] = []
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunks.append(CaptionChunk(words=chunk_words))
        return chunks

    def calculate_layouts(self, chunks: List[CaptionChunk], video_width: int, video_height: Optional[int] = None) -> None:
        """Precompute optimal font size and word widths for each chunk."""
        meas_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bg_pad_x = self.style.chunk_bg_padding[0] if self.style.chunk_bg else 0
        max_allowed_width = max(100, video_width - 80 - bg_pad_x * 2)

        # Spacing multiplier: extra breathing room for bounding box pills or word bounce
        if self.style.active_box_highlight:
            space_factor = 1.35
        elif self.style.word_bounce:
            space_factor = 1.20
        else:
            space_factor = 1.05

        is_widescreen = bool(video_height and video_width > video_height)
        if is_widescreen:
            # Scale typography for 16:9 widescreen canvas
            max_size = min(46, max(24, int(round(video_height * 0.045))))
            min_size = max(18, int(round(max_size * 0.55)))
            size_range = range(max_size, min_size - 1, -2)
        else:
            # 9:16 vertical canvas default
            size_range = range(64, 31, -4)

        for chunk in chunks:
            best_meta: Optional[ChunkLayoutMeta] = None
            # Test decreasing font sizes until chunk fits horizontally
            for size in size_range:
                font = self.font_mgr.get_font(size, font_family=self.style.font_family)
                space_w = meas_draw.textlength(" ", font=font) * space_factor
                word_widths = [
                    meas_draw.textlength(w.word.upper() if self.style.uppercase else w.word, font=font)
                    for w in chunk.words
                ]
                total_w = sum(word_widths) + space_w * max(0, len(chunk.words) - 1)
                best_meta = ChunkLayoutMeta(
                    font_size=size,
                    space_width=space_w,
                    word_widths=word_widths,
                    total_width=total_w
                )
                if total_w <= max_allowed_width:
                    break
            chunk.meta = best_meta

    def _render_sprite(
        self,
        chunk: CaptionChunk,
        active_word_idx: int,
        width: int,
        height: int,
        y_pos: int,
    ) -> Optional[CaptionSprite]:
        """
        Pre-render a single (chunk, active_word_idx) state into a tight RGBA sprite
        with premultiplied alpha and bounding coordinates on the target canvas.
        """
        if not chunk.meta:
            return None

        meta = chunk.meta
        base_font = self.font_mgr.get_font(meta.font_size, font_family=self.style.font_family)
        active_font = (
            self.font_mgr.get_font(int(meta.font_size * self.style.bounce_scale), font_family=self.style.font_family)
            if self.style.word_bounce
            else base_font
        )

        x_start = (width - meta.total_width) / 2.0
        pad_x, pad_y = self.style.chunk_bg_padding if self.style.chunk_bg else (0, 0)
        pad_bx, pad_by = self.style.active_box_padding if self.style.active_box_highlight else (0, 0)

        glow_extra = (self.style.glow_radius * 4 + self.style.glow_stroke + 20) if self.style.glow else 0
        shadow_extra_x = max(
            abs(self.style.shadow_offset[0]) if self.style.shadow_offset else 0,
            self.style.shadow_3d_depth if self.style.shadow_3d else 0
        )
        shadow_extra_y = max(
            abs(self.style.shadow_offset[1]) if self.style.shadow_offset else 0,
            self.style.shadow_3d_depth if self.style.shadow_3d else 0
        )
        stroke_extra = self.style.stroke_width + 4
        bounce_extra = int(meta.font_size * abs(self.style.bounce_scale - 1.0)) + abs(self.style.bounce_y_offset) if self.style.word_bounce else 0

        margin_x = max(pad_x, pad_bx) + shadow_extra_x + stroke_extra + glow_extra + bounce_extra + 16
        margin_y = max(pad_y, pad_by) + shadow_extra_y + stroke_extra + glow_extra + bounce_extra + 16

        meas_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        sample_box = meas_draw.textbbox((0, 0), "Ag", font=base_font)
        line_h = sample_box[3] - sample_box[1]

        roi_x0 = max(0, int(np.floor(x_start - margin_x)))
        roi_x1 = min(width, int(np.ceil(x_start + meta.total_width + margin_x)))
        roi_y0 = max(0, int(np.floor(y_pos - margin_y)))
        roi_y1 = min(height, int(np.ceil(y_pos + line_h + margin_y)))
        sw = roi_x1 - roi_x0
        sh = roi_y1 - roi_y0
        if sw <= 0 or sh <= 0:
            return None

        overlay = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        glow_overlay = Image.new("RGBA", (sw, sh), (0, 0, 0, 0)) if self.style.glow else None
        glow_draw = ImageDraw.Draw(glow_overlay) if glow_overlay else None

        x_start_s = x_start - roi_x0
        y_pos_s = y_pos - roi_y0

        # A. Chunk container pill backdrop
        if self.style.chunk_bg:
            c_rect = [
                x_start_s - pad_x,
                y_pos_s - pad_y,
                x_start_s + meta.total_width + pad_x,
                y_pos_s + line_h + pad_y + 4
            ]
            draw.rounded_rectangle(
                c_rect,
                radius=self.style.chunk_bg_radius,
                fill=self.style.chunk_bg_fill,
                outline=self.style.chunk_bg_border,
                width=1
            )

        # B. Render words
        x_cursor = x_start_s
        for idx, w in enumerate(chunk.words):
            txt = w.word.upper() if self.style.uppercase else w.word
            # Strict invariant asserting Latin characters only
            assert not DEV.search(txt) and txt.isascii(), f"Non-Latin token in renderer: {txt!r}"

            slot_w = meta.word_widths[idx]
            x_center = x_cursor + slot_w / 2.0
            is_active = (idx == active_word_idx)
            is_past = (active_word_idx != -1 and idx < active_word_idx)
            is_upcoming = (active_word_idx != -1 and idx > active_word_idx)

            if is_active:
                font = active_font
                word_len = draw.textlength(txt, font=font)
                x_draw = x_center - word_len / 2.0
                y_draw = y_pos_s + (self.style.bounce_y_offset if self.style.word_bounce else 0)

                if self.style.active_box_highlight:
                    bbox = draw.textbbox((x_draw, y_draw), txt, font=font)
                    box_coords = [
                        bbox[0] - pad_bx,
                        bbox[1] - pad_by,
                        bbox[2] + pad_bx,
                        bbox[3] + pad_by
                    ]
                    draw.rounded_rectangle(
                        box_coords,
                        radius=self.style.active_box_radius,
                        fill=self.style.active_box_fill
                    )
                    draw.text(
                        (x_draw, y_draw),
                        txt,
                        font=font,
                        fill=(*self.style.active_box_text_color, 255)
                    )
                else:
                    if glow_draw:
                        glow_draw.text(
                            (x_draw, y_draw),
                            txt,
                            font=font,
                            fill=(*self.style.glow_color, 220),
                            stroke_width=self.style.glow_stroke,
                            stroke_fill=(*self.style.glow_color, 220)
                        )

                    if self.style.shadow_3d:
                        for d in range(1, self.style.shadow_3d_depth + 1):
                            draw.text(
                                (x_draw + int(d * 0.75), y_draw + d),
                                txt,
                                font=font,
                                fill=(*self.style.shadow_3d_color, 255),
                                stroke_width=self.style.stroke_width,
                                stroke_fill=(*self.style.shadow_3d_color, 255)
                            )
                    elif self.style.shadow_offset:
                        sx = x_draw + self.style.shadow_offset[0]
                        sy = y_draw + self.style.shadow_offset[1]
                        draw.text(
                            (sx, sy),
                            txt,
                            font=font,
                            fill=self.style.shadow_color,
                            stroke_width=self.style.stroke_width,
                            stroke_fill=self.style.shadow_color
                        )

                    draw.text(
                        (x_draw, y_draw),
                        txt,
                        font=font,
                        fill=(*self.style.active_color, 255),
                        stroke_width=self.style.stroke_width,
                        stroke_fill=(*self.style.stroke_color, 255)
                    )
            else:
                font = base_font
                x_draw = x_cursor
                y_draw = y_pos_s
                base_alpha = self.style.future_alpha if is_upcoming else self.style.past_alpha

                if glow_draw:
                    g_alpha = int(60 * (base_alpha / 255.0))
                    glow_draw.text(
                        (x_draw, y_draw),
                        txt,
                        font=font,
                        fill=(255, 255, 255, g_alpha),
                        stroke_width=3,
                        stroke_fill=(255, 255, 255, g_alpha)
                    )

                if self.style.shadow_offset:
                    sx = x_draw + self.style.shadow_offset[0]
                    sy = y_draw + self.style.shadow_offset[1]
                    s_alpha = int(self.style.shadow_color[3] * (base_alpha / 255.0))
                    draw.text(
                        (sx, sy),
                        txt,
                        font=font,
                        fill=(*self.style.shadow_color[:3], s_alpha),
                        stroke_width=self.style.stroke_width,
                        stroke_fill=(*self.style.shadow_color[:3], s_alpha)
                    )
                draw.text(
                    (x_draw, y_draw),
                    txt,
                    font=font,
                    fill=(*self.style.inactive_color, base_alpha),
                    stroke_width=self.style.stroke_width,
                    stroke_fill=(*self.style.stroke_color, base_alpha)
                )

            x_cursor += slot_w + meta.space_width

        # C. Composite glow inside sprite bounding box
        if glow_overlay:
            glow_blurred = glow_overlay.filter(ImageFilter.GaussianBlur(radius=self.style.glow_radius))
            sprite_pil = Image.alpha_composite(glow_blurred, overlay)
        else:
            sprite_pil = overlay

        # D. Tighten bounding box
        sprite_arr = np.array(sprite_pil)
        alpha_chan = sprite_arr[:, :, 3]
        if not np.any(alpha_chan > 0):
            return None

        rows = np.any(alpha_chan > 0, axis=1)
        cols = np.any(alpha_chan > 0, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        tight_x0 = roi_x0 + int(xmin)
        tight_x1 = roi_x0 + int(xmax) + 1
        tight_y0 = roi_y0 + int(ymin)
        tight_y1 = roi_y0 + int(ymax) + 1

        tight_arr = sprite_arr[ymin:ymax + 1, xmin:xmax + 1]
        sprite_bgr_f = tight_arr[:, :, [2, 1, 0]].astype(np.float32)
        alpha_f = tight_arr[:, :, 3:4].astype(np.float32) / 255.0
        sprite_pre = sprite_bgr_f * alpha_f
        inv_alpha = 1.0 - alpha_f

        return CaptionSprite(
            roi_x0=tight_x0,
            roi_y0=tight_y0,
            roi_x1=tight_x1,
            roi_y1=tight_y1,
            sprite_pre=sprite_pre,
            inv_alpha=inv_alpha,
            alpha_f=alpha_f,
            sprite_bgr_f=sprite_bgr_f,
        )

    def _render_frame_legacy(
        self,
        frame: np.ndarray,
        active_chunk: CaptionChunk,
        active_word_idx: int,
        current_time: float,
        width: int,
        height: int,
        y_pos: int,
        needs_alpha: bool,
    ) -> np.ndarray:
        """Reference legacy frame-by-frame PIL drawing implementation (for pixel verification)."""
        meta = active_chunk.meta
        if not meta:
            return frame
        base_font = self.font_mgr.get_font(meta.font_size, font_family=self.style.font_family)

        if needs_alpha:
            pil_base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            glow_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0)) if self.style.glow else None
            glow_draw = ImageDraw.Draw(glow_overlay) if glow_overlay else None

            fade_mult = 1.0
            if self.style.chunk_fade_in:
                dt = current_time - active_chunk.start
                if 0.0 <= dt < self.style.chunk_fade_duration:
                    fade_mult = max(0.1, min(1.0, dt / self.style.chunk_fade_duration))

            x_start = (width - meta.total_width) / 2.0

            if self.style.chunk_bg:
                sample_box = draw.textbbox((0, 0), "Ag", font=base_font)
                line_h = sample_box[3] - sample_box[1]
                pad_x, pad_y = self.style.chunk_bg_padding
                c_rect = [
                    x_start - pad_x,
                    y_pos - pad_y,
                    x_start + meta.total_width + pad_x,
                    y_pos + line_h + pad_y + 4
                ]
                bg_fill = self.style.chunk_bg_fill
                if fade_mult < 1.0:
                    bg_fill = (*bg_fill[:3], int(bg_fill[3] * fade_mult))
                border = self.style.chunk_bg_border
                if border and fade_mult < 1.0:
                    border = (*border[:3], int(border[3] * fade_mult))
                draw.rounded_rectangle(
                    c_rect,
                    radius=self.style.chunk_bg_radius,
                    fill=bg_fill,
                    outline=border,
                    width=1
                )

            active_font = (
                self.font_mgr.get_font(int(meta.font_size * self.style.bounce_scale), font_family=self.style.font_family)
                if self.style.word_bounce
                else base_font
            )

            x_cursor = x_start
            for idx, w in enumerate(active_chunk.words):
                txt = w.word.upper() if self.style.uppercase else w.word
                assert not DEV.search(txt) and txt.isascii(), f"Non-Latin token in renderer: {txt!r}"

                slot_w = meta.word_widths[idx]
                x_center = x_cursor + slot_w / 2.0
                is_active = (idx == active_word_idx)
                is_past = (active_word_idx != -1 and idx < active_word_idx)
                is_upcoming = (active_word_idx != -1 and idx > active_word_idx)

                if is_active:
                    font = active_font
                    word_len = draw.textlength(txt, font=font)
                    x_draw = x_center - word_len / 2.0
                    y_draw = y_pos + (self.style.bounce_y_offset if self.style.word_bounce else 0)

                    if self.style.active_box_highlight:
                        pad_bx, pad_by = self.style.active_box_padding
                        bbox = draw.textbbox((x_draw, y_draw), txt, font=font)
                        box_coords = [
                            bbox[0] - pad_bx,
                            bbox[1] - pad_by,
                            bbox[2] + pad_bx,
                            bbox[3] + pad_by
                        ]
                        box_fill = self.style.active_box_fill
                        if fade_mult < 1.0:
                            box_fill = (*box_fill[:3], int(box_fill[3] * fade_mult))
                        draw.rounded_rectangle(
                            box_coords,
                            radius=self.style.active_box_radius,
                            fill=box_fill
                        )
                        alpha = int(255 * fade_mult)
                        draw.text(
                            (x_draw, y_draw),
                            txt,
                            font=font,
                            fill=(*self.style.active_box_text_color, alpha)
                        )
                    else:
                        if glow_draw:
                            g_alpha = int(220 * fade_mult)
                            glow_draw.text(
                                (x_draw, y_draw),
                                txt,
                                font=font,
                                fill=(*self.style.glow_color, g_alpha),
                                stroke_width=self.style.glow_stroke,
                                stroke_fill=(*self.style.glow_color, g_alpha)
                            )

                        if self.style.shadow_3d:
                            s_alpha = int(255 * fade_mult)
                            for d in range(1, self.style.shadow_3d_depth + 1):
                                draw.text(
                                    (x_draw + int(d * 0.75), y_draw + d),
                                    txt,
                                    font=font,
                                    fill=(*self.style.shadow_3d_color, s_alpha),
                                    stroke_width=self.style.stroke_width,
                                    stroke_fill=(*self.style.shadow_3d_color, s_alpha)
                                )
                        elif self.style.shadow_offset:
                            sx = x_draw + self.style.shadow_offset[0]
                            sy = y_draw + self.style.shadow_offset[1]
                            s_alpha = int(self.style.shadow_color[3] * fade_mult)
                            draw.text(
                                (sx, sy),
                                txt,
                                font=font,
                                fill=(*self.style.shadow_color[:3], s_alpha),
                                stroke_width=self.style.stroke_width,
                                stroke_fill=(*self.style.shadow_color[:3], s_alpha)
                            )

                        alpha = int(255 * fade_mult)
                        draw.text(
                            (x_draw, y_draw),
                            txt,
                            font=font,
                            fill=(*self.style.active_color, alpha),
                            stroke_width=self.style.stroke_width,
                            stroke_fill=(*self.style.stroke_color, alpha)
                        )
                else:
                    font = base_font
                    x_draw = x_cursor
                    y_draw = y_pos
                    base_alpha = self.style.future_alpha if is_upcoming else self.style.past_alpha
                    alpha = int(base_alpha * fade_mult)

                    if glow_draw:
                        g_alpha = int(60 * (alpha / 255.0))
                        glow_draw.text(
                            (x_draw, y_draw),
                            txt,
                            font=font,
                            fill=(255, 255, 255, g_alpha),
                            stroke_width=3,
                            stroke_fill=(255, 255, 255, g_alpha)
                        )

                    if self.style.shadow_offset:
                        sx = x_draw + self.style.shadow_offset[0]
                        sy = y_draw + self.style.shadow_offset[1]
                        s_alpha = int(self.style.shadow_color[3] * (alpha / 255.0))
                        draw.text(
                            (sx, sy),
                            txt,
                            font=font,
                            fill=(*self.style.shadow_color[:3], s_alpha),
                            stroke_width=self.style.stroke_width,
                            stroke_fill=(*self.style.shadow_color[:3], s_alpha)
                        )
                    draw.text(
                        (x_draw, y_draw),
                        txt,
                        font=font,
                        fill=(*self.style.inactive_color, alpha),
                        stroke_width=self.style.stroke_width,
                        stroke_fill=(*self.style.stroke_color, alpha)
                    )

                x_cursor += slot_w + meta.space_width

            pil_rgba = pil_base.convert("RGBA")
            if glow_overlay:
                glow_blurred = glow_overlay.filter(ImageFilter.GaussianBlur(radius=self.style.glow_radius))
                pil_rgba = Image.alpha_composite(pil_rgba, glow_blurred)
            pil_rgba = Image.alpha_composite(pil_rgba, overlay)
            return cv2.cvtColor(np.array(pil_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)

        else:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            x_cursor = (width - meta.total_width) / 2.0
            for idx, w in enumerate(active_chunk.words):
                txt = w.word.upper() if self.style.uppercase else w.word
                assert not DEV.search(txt) and txt.isascii(), f"Non-Latin token in renderer: {txt!r}"

                fill_color = self.active_color if idx == active_word_idx else self.inactive_color
                draw.text(
                    (x_cursor, y_pos),
                    txt,
                    font=base_font,
                    fill=fill_color,
                    stroke_width=self.stroke_width,
                    stroke_fill=self.stroke_color
                )
                x_cursor += meta.word_widths[idx] + meta.space_width

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def create_composer(
        self,
        words: List[WordCaption],
        width: int,
        height: int,
        legacy: bool = False,
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Create a per-frame caption compositor callable.
        Takes (frame: np.ndarray [BGR], current_time: float) -> np.ndarray [BGR].

        When legacy=False (default):
        - Pre-renders each (chunk, active_word_idx) state ONCE as an RGBA sprite covering
          only the text bounding box plus padding.
        - Per frame, blends only the sprite ROI into the frame with premultiplied alpha
          in NumPy (no full-frame PIL conversions, no full-frame alpha composite).
        - Frames with no active caption pass through untouched (0 color conversions).

        When legacy=True:
        - Uses the original frame-by-frame PIL drawing logic (for pixel verification).
        """
        chunks = self.build_chunks(words)
        self.calculate_layouts(chunks, width, height)

        is_widescreen = (width > height)
        if is_widescreen:
            if self.bottom_offset == 380:
                effective_bottom_offset = min(120, max(45, int(height * 0.11)))
            else:
                effective_bottom_offset = self.bottom_offset
        elif height <= 1450 and width < height:
            if self.bottom_offset == 380:
                effective_bottom_offset = int(height * 0.18)
            else:
                effective_bottom_offset = self.bottom_offset
        else:
            effective_bottom_offset = self.bottom_offset

        y_pos = max(30, height - effective_bottom_offset + self.style.bottom_offset_adjust)

        if legacy:
            needs_alpha = (
                self.style.chunk_bg
                or self.style.active_box_highlight
                or self.style.shadow_offset is not None
                or self.style.future_alpha < 255
                or self.style.chunk_fade_in
                or self.style.glow
                or self.style.shadow_3d
            )

            def legacy_compose(frame: np.ndarray, current_time: float) -> np.ndarray:
                if not chunks:
                    return frame

                active_chunk: Optional[CaptionChunk] = None
                active_word_idx = -1

                for chunk in chunks:
                    if chunk.start <= current_time <= chunk.end + 0.20:
                        active_chunk = chunk
                        for idx, w in enumerate(chunk.words):
                            if w.start <= current_time <= w.end:
                                active_word_idx = idx
                                break
                        if active_word_idx == -1 and chunk.words:
                            for idx, w in enumerate(chunk.words):
                                if current_time >= w.start:
                                    active_word_idx = idx
                        break

                if not active_chunk or not active_chunk.meta:
                    return frame

                return self._render_frame_legacy(
                    frame=frame,
                    active_chunk=active_chunk,
                    active_word_idx=active_word_idx,
                    current_time=current_time,
                    width=width,
                    height=height,
                    y_pos=y_pos,
                    needs_alpha=needs_alpha,
                )

            return legacy_compose

        # ── Pre-Render Sprites Path (Ultra-Fast NumPy Premultiplied Alpha Blend) ──
        sprites: Dict[Tuple[int, int], CaptionSprite] = {}
        for c_idx, chunk in enumerate(chunks):
            if not chunk.meta:
                continue
            for a_idx in range(-1, len(chunk.words)):
                sp = self._render_sprite(chunk, a_idx, width, height, y_pos)
                if sp is not None:
                    sprites[(c_idx, a_idx)] = sp

        chunk_fade_in = self.style.chunk_fade_in
        chunk_fade_dur = self.style.chunk_fade_duration

        def compose(frame: np.ndarray, current_time: float) -> np.ndarray:
            if not chunks:
                return frame

            # Locate active chunk for current timestamp
            active_chunk_idx = -1
            active_word_idx = -1
            active_chunk_start = 0.0

            for c_idx, chunk in enumerate(chunks):
                if chunk.start <= current_time <= chunk.end + 0.20:
                    active_chunk_idx = c_idx
                    active_chunk_start = chunk.start
                    for idx, w in enumerate(chunk.words):
                        if w.start <= current_time <= w.end:
                            active_word_idx = idx
                            break
                    if active_word_idx == -1 and chunk.words:
                        for idx, w in enumerate(chunk.words):
                            if current_time >= w.start:
                                active_word_idx = idx
                    break

            if active_chunk_idx == -1:
                # Untouched pass-through when no caption is active
                return frame

            sprite = sprites.get((active_chunk_idx, active_word_idx))
            if sprite is None or sprite.sprite_pre is None:
                return frame

            y0, y1 = sprite.roi_y0, sprite.roi_y1
            x0, x1 = sprite.roi_x0, sprite.roi_x1

            if chunk_fade_in:
                dt = current_time - active_chunk_start
                if 0.0 <= dt < chunk_fade_dur:
                    fade_mult = max(0.1, min(1.0, dt / chunk_fade_dur))
                    eff_alpha = sprite.alpha_f * fade_mult
                    eff_inv_alpha = 1.0 - eff_alpha
                    eff_sprite_pre = sprite.sprite_bgr_f * eff_alpha
                    roi = frame[y0:y1, x0:x1]
                    frame[y0:y1, x0:x1] = np.clip(
                        np.round(roi.astype(np.float32) * eff_inv_alpha + eff_sprite_pre),
                        0, 255
                    ).astype(np.uint8)
                    return frame

            roi = frame[y0:y1, x0:x1]
            frame[y0:y1, x0:x1] = np.clip(
                np.round(roi.astype(np.float32) * sprite.inv_alpha + sprite.sprite_pre),
                0, 255
            ).astype(np.uint8)
            return frame

        return compose

    def render_to_video(
        self,
        input_video: Path | str,
        output_video: Path | str,
        words: List[WordCaption]
    ) -> Path:
        """
        Stream frames from input_video, composite word captions using Pillow,
        and encode to output_video via FFmpeg rawvideo pipe.
        """
        in_path = Path(input_video).expanduser().resolve()
        out_path = Path(output_video).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure input is Constant Frame Rate (CFR) and in AV sync
        in_path = ensure_cfr_video(in_path)

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video for caption burning: {in_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[caption] Input: {width}x{height} @ {fps:.2f}fps ({total_frames} frames)")
        print(f"[caption] Style: '{self.style.name}' ({self.style.description})")

        compose = self.create_composer(words, width, height)

        is_widescreen = (width > height)
        aspect_desc = '16:9' if is_widescreen else ('4:5' if height <= 1450 else '9:16')
        print(f"[caption] Burning captions (aspect={aspect_desc})...")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-i", str(in_path),
            "-map", "0:v", "-map", "1:a:0",
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0",
            "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", str(out_path)
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                current_time = frame_idx / fps
                frame_idx += 1

                frame = compose(frame, current_time)

                if proc.stdin:
                    proc.stdin.write(frame.tobytes())
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg caption encoding failed with return code {proc.returncode}")

        verify_av_sync(out_path, max_diff=0.05)
        print(f"[caption] Captions successfully burned -> {out_path}")
        return out_path


def burn_captions(
    input_video: Path | str,
    output_video: Path | str,
    words: List[WordCaption],
    chunk_size: int = 4,
    bottom_offset: int = 380,
    font_path: Optional[str | Path] = None,
    style: Optional[Union[str, CaptionStyle]] = "classic-yellow",
) -> Path:
    """Convenience functional interface for burning captions."""
    renderer = CaptionRenderer(
        font_path=font_path,
        chunk_size=chunk_size,
        bottom_offset=bottom_offset,
        style=style
    )
    return renderer.render_to_video(input_video, output_video, words)
