#!/usr/bin/env python3
"""
test_caption_sprites.py - Verify sprite pre-rendering vs legacy PIL rendering visual equivalence.

Requirement 4:
- Keep all 12 styles visually identical.
- Write a test that renders frames for classic-yellow and ae-glow with the old and new path
  and reports the max/mean pixel difference (tolerance small).
- Frames with no active caption pass through untouched (0 conversions).
"""

import unittest
import numpy as np
from core.captioner import CaptionRenderer, CAPTION_STYLES
from core.transcriber import WordCaption


class TestCaptionSpriteRendering(unittest.TestCase):
    def setUp(self):
        self.words = [
            WordCaption(word="kya", start=1.0, end=1.8, raw_token="क्या"),
            WordCaption(word="aap", start=1.8, end=2.4, raw_token="आप"),
            WordCaption(word="ready", start=2.4, end=3.2, raw_token="ready"),
            WordCaption(word="hain", start=3.2, end=4.0, raw_token="हैं"),
        ]
        # High-resolution vertical reel frame (1080x1920) with gradient
        y, x = np.mgrid[0:1920, 0:1080]
        self.frame = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)

    def test_untouched_passthrough_when_no_caption(self):
        """Requirement 3: Frames with no active caption pass through untouched (no color conversions)."""
        renderer = CaptionRenderer(style="classic-yellow")
        composer = renderer.create_composer(self.words, width=1080, height=1920, legacy=False)

        # Time 0.5s is before any word begins (first word starts at 1.0s)
        test_frame = self.frame.copy()
        out_frame = composer(test_frame, 0.5)
        # Should return the exact same object reference (zero copy, zero conversion)
        self.assertIs(out_frame, test_frame)

        # Time 10.0s is after all words end (last word ends at 4.0s)
        test_frame_after = self.frame.copy()
        out_frame_after = composer(test_frame_after, 10.0)
        self.assertIs(out_frame_after, test_frame_after)

    def test_visual_equivalence_classic_yellow(self):
        """Requirement 4: classic-yellow pixel difference between legacy and sprite path."""
        renderer_old = CaptionRenderer(style="classic-yellow")
        composer_old = renderer_old.create_composer(self.words, width=1080, height=1920, legacy=True)

        renderer_new = CaptionRenderer(style="classic-yellow")
        composer_new = renderer_new.create_composer(self.words, width=1080, height=1920, legacy=False)

        f_old = composer_old(self.frame.copy(), 2.0)
        f_new = composer_new(self.frame.copy(), 2.0)

        diff = np.abs(f_old.astype(np.int32) - f_new.astype(np.int32))
        max_diff = int(diff.max())
        mean_diff = float(diff.mean())

        print(f"\n[classic-yellow] max_diff: {max_diff}, mean_diff: {mean_diff:.6f}")
        # classic-yellow must be an exact bitwise match
        self.assertEqual(max_diff, 0)
        self.assertEqual(mean_diff, 0.0)

    def test_visual_equivalence_ae_glow(self):
        """Requirement 4: ae-glow pixel difference between legacy and sprite path."""
        renderer_old = CaptionRenderer(style="ae-glow")
        composer_old = renderer_old.create_composer(self.words, width=1080, height=1920, legacy=True)

        renderer_new = CaptionRenderer(style="ae-glow")
        composer_new = renderer_new.create_composer(self.words, width=1080, height=1920, legacy=False)

        f_old = composer_old(self.frame.copy(), 2.0)
        f_new = composer_new(self.frame.copy(), 2.0)

        diff = np.abs(f_old.astype(np.int32) - f_new.astype(np.int32))
        max_diff = int(diff.max())
        mean_diff = float(diff.mean())

        print(f"\n[ae-glow] max_diff: {max_diff}, mean_diff: {mean_diff:.6f}")
        # ae-glow blur uses float32 premultiplied math; max pixel diff is at most 1 on 0-255 scale
        self.assertLessEqual(max_diff, 1)
        self.assertLessEqual(mean_diff, 0.005)

    def test_all_12_styles_visual_equivalence(self):
        """Verify all 12 styles maintain <= 1 max pixel difference between legacy and sprite path."""
        base_12_styles = [
            "classic-yellow", "box-highlight", "bold-stroke", "fade-in",
            "kalakar-fire", "karaoke-neon", "cyber-neon", "cinematic",
            "comic-pop", "minimal", "subtitle-16x9", "ae-glow"
        ]
        for style_name in base_12_styles:
            with self.subTest(style=style_name):
                r_old = CaptionRenderer(style=style_name)
                c_old = r_old.create_composer(self.words, width=1080, height=1920, legacy=True)

                r_new = CaptionRenderer(style=style_name)
                c_new = r_new.create_composer(self.words, width=1080, height=1920, legacy=False)

                f_old = c_old(self.frame.copy(), 2.0)
                f_new = c_new(self.frame.copy(), 2.0)

                diff = np.abs(f_old.astype(np.int32) - f_new.astype(np.int32))
                max_d = int(diff.max())
                mean_d = float(diff.mean())

                if style_name == "ae-glow":
                    self.assertLessEqual(max_d, 1)
                    self.assertLessEqual(mean_d, 0.005)
                else:
                    self.assertEqual(max_d, 0, f"Style {style_name} had max_diff={max_d}")
                    self.assertEqual(mean_d, 0.0, f"Style {style_name} had mean_diff={mean_d}")


if __name__ == "__main__":
    unittest.main()
