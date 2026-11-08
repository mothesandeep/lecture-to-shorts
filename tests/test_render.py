#!/usr/bin/env python3
"""
test_render.py - Unit tests for ReframePlan, CaptionRenderer composer, and render_reel invariants.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from core.reframer import FaceReframer, ReframeConfig, ReframePlan
from core.captioner import CaptionRenderer, CaptionStyle
from core.transcriber import WordCaption, sanitize_latin


class TestRenderPlanAndComposer(unittest.TestCase):
    def test_caption_composer_latin_enforcement(self):
        """Verify compose callable enforces Latin ASCII invariant."""
        renderer = CaptionRenderer(chunk_size=3)
        valid_words = [
            WordCaption(word="kya", start=0.0, end=1.0, raw_token="क्या"),
            WordCaption(word="aap", start=1.0, end=2.0, raw_token="आप"),
            WordCaption(word="ready", start=2.0, end=3.0, raw_token="ready"),
        ]
        composer = renderer.create_composer(valid_words, width=1080, height=1920)

        # Create dummy black frame
        dummy_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

        # Should render without error
        out_frame = composer(dummy_frame, 0.5)
        self.assertEqual(out_frame.shape, (1920, 1080, 3))
        # Ensure frame was drawn on (not all zeros)
        self.assertGreater(out_frame.sum(), 0)

        # Now test with un-sanitized Devanagari token -> should trigger AssertionError
        invalid_words = [
            WordCaption(word="क्या", start=0.0, end=1.0, raw_token="क्या"),
        ]
        # In sprite pre-rendering mode, assertion is enforced at sprite creation time
        with self.assertRaises(AssertionError):
            renderer.create_composer(invalid_words, width=1080, height=1920)

        # In legacy mode, assertion is enforced at frame render time
        legacy_composer = renderer.create_composer(invalid_words, width=1080, height=1920, legacy=True)
        with self.assertRaises(AssertionError):
            legacy_composer(dummy_frame, 0.5)

    def test_caption_composer_outside_timestamps(self):
        """Verify compose returns original frame unmodified when outside caption intervals."""
        renderer = CaptionRenderer(chunk_size=2)
        words = [
            WordCaption(word="hello", start=5.0, end=6.0, raw_token="hello"),
        ]
        composer = renderer.create_composer(words, width=1080, height=1920)
        dummy_frame = np.ones((1920, 1080, 3), dtype=np.uint8) * 42

        # Timestamp 1.0s is before the first chunk (starts at 5.0)
        out_frame = composer(dummy_frame, 1.0)
        self.assertTrue(np.array_equal(out_frame, dummy_frame))

    def test_reframe_plan_fields(self):
        """Verify ReframePlan stores all required geometry and trajectory attributes."""
        plan = ReframePlan(
            src_path=Path("dummy_src.mp4"),
            out_path=Path("dummy_out.mp4"),
            width=1920,
            height=1080,
            fps=25.0,
            total_frames=100,
            num_frames=100,
            crop_w=607,
            crop_h=1080,
            out_canvas_w=1080,
            out_canvas_h=1920,
            crop_x0=np.zeros(100, dtype=np.float32),
            crop_y0=np.zeros(100, dtype=np.float32),
            scene_cuts=[0, 100],
            cut_indices={0, 100},
            layout="direct-crop",
            is_4_5=False,
            zoom=1.0,
            preset_name="talking_head",
            headroom=0.18,
            encoder_preset="fast",
            crf=17,
            audio_bitrate="192k",
        )
        self.assertEqual(plan.out_canvas_w, 1080)
        self.assertEqual(plan.out_canvas_h, 1920)
        self.assertEqual(len(plan.crop_x0), 100)


if __name__ == "__main__":
    unittest.main()
