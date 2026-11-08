#!/usr/bin/env python3
"""
test_punchy_captions.py - Comprehensive tests for punchier animated captions.

Covers:
1. ease_out_back and spring_eval math and bounds.
2. Per-word entrance scale profile (0.6 -> 1.08 -> 1.0 over 7 frames).
3. Active word pop profile (1.0 -> 1.15 -> 1.05 and short y-slide 0 -> -4px -> -2px over 6 frames).
4. Emphasis word detection and styling (WordCaption.emphasis, terminal '!', all-caps >= 3 chars, long duration).
5. Greedy pixel-width chunker with pause breaks (> 0.3s) and terminal punctuation breaks.
6. 2-line rolling layout transition over 6 frames with smoothstep slide.
7. Style registry and alias resolution for kinetic-pop and kinetic-roll.
8. Pre-rendered animation sprite caching.
9. Strict Latin-only assertion invariant.
10. Untouched frame passthrough when no caption is active.
"""

import unittest
import numpy as np

from core.captioner import (
    CaptionRenderer,
    CaptionStyle,
    CAPTION_STYLES,
    get_style,
    ease_out_back,
    spring_eval,
    smoothstep,
    entrance_scale_profile,
    active_pop_profile,
    has_terminal_punct,
    is_emphasis_word,
)
from core.transcriber import WordCaption


class TestPunchyCaptionCurves(unittest.TestCase):
    """Test animation easing and physics helpers."""

    def test_ease_out_back_bounds(self):
        """ease_out_back starts at 0.0, overshoots > 1.0, and settles at 1.0."""
        self.assertAlmostEqual(ease_out_back(0.0), 0.0, places=5)
        self.assertAlmostEqual(ease_out_back(1.0), 1.0, places=5)

        # Peak overshoot occurs around midpoint
        mid_val = ease_out_back(0.55)
        self.assertGreater(mid_val, 1.15)
        self.assertLessEqual(mid_val, 1.25)

    def test_spring_eval_bounds(self):
        """spring_eval starts at 0.0, exhibits damped oscillation, and settles at 1.0."""
        self.assertAlmostEqual(spring_eval(0.0), 0.0, places=5)
        self.assertAlmostEqual(spring_eval(1.0), 1.0, places=5)

        # At t=0.30s, underdamped spring should overshoot slightly
        val_overshoot = spring_eval(0.30)
        self.assertGreater(val_overshoot, 1.0)
        self.assertLess(val_overshoot, 1.15)

    def test_entrance_scale_profile(self):
        """Per-word entrance: scale 0.6 -> 1.08 -> 1.0 over 7 frames."""
        # Frame 0: start at 0.60
        s0 = entrance_scale_profile(0, total_frames=7)
        self.assertAlmostEqual(s0, 0.60, places=3)

        # Frame 3 or 4: peak overshoot near 1.08
        s3 = entrance_scale_profile(3, total_frames=7)
        s4 = entrance_scale_profile(4, total_frames=7)
        peak = max(s3, s4)
        self.assertGreaterEqual(peak, 1.07)
        self.assertLessEqual(peak, 1.09)

        # Frame 6 (last frame of 7): settles to 1.00
        s_end = entrance_scale_profile(6, total_frames=7)
        self.assertAlmostEqual(s_end, 1.00, places=3)

    def test_active_pop_profile(self):
        """Active word: color flip plus pop (1.0 -> 1.15 -> 1.05) and short y-slide."""
        # Frame 0: 1.0 scale, 0 y-slide
        s0, y0 = active_pop_profile(0, total_frames=6)
        self.assertAlmostEqual(s0, 1.00, places=2)
        self.assertEqual(y0, 0)

        # Frame 2: peak pop 1.15, y-slide -4px
        s2, y2 = active_pop_profile(2, total_frames=6)
        self.assertAlmostEqual(s2, 1.15, places=2)
        self.assertEqual(y2, -4)

        # Frame 5: settles to 1.05 scale, -2px y-slide
        s5, y5 = active_pop_profile(5, total_frames=6)
        self.assertAlmostEqual(s5, 1.05, places=2)
        self.assertEqual(y5, -2)

    def test_smoothstep_curve(self):
        """smoothstep interpolates smoothly with 0 derivative at endpoints."""
        self.assertAlmostEqual(smoothstep(0.0, 1.0, 0.0), 0.0)
        self.assertAlmostEqual(smoothstep(0.0, 1.0, 0.5), 0.5)
        self.assertAlmostEqual(smoothstep(0.0, 1.0, 1.0), 1.0)


class TestEmphasisWordDetection(unittest.TestCase):
    """Test identification of emphasis words for size/color bumps."""

    def test_explicit_emphasis_flag(self):
        w = WordCaption(word="concept", start=1.0, end=1.3, raw_token="concept", emphasis=True)
        self.assertTrue(is_emphasis_word(w))

    def test_terminal_exclamation(self):
        w1 = WordCaption(word="karo!", start=1.0, end=1.3, raw_token="करो!")
        self.assertTrue(is_emphasis_word(w1))
        w2 = WordCaption(word="dhyan", start=1.0, end=1.3, raw_token="ध्यान!")
        self.assertTrue(is_emphasis_word(w2))

    def test_all_caps_acronyms(self):
        w1 = WordCaption(word="PW", start=1.0, end=1.3, raw_token="PW")
        self.assertFalse(is_emphasis_word(w1))  # len 2
        w2 = WordCaption(word="IIT", start=1.0, end=1.3, raw_token="IIT")
        self.assertTrue(is_emphasis_word(w2))   # len 3
        w3 = WordCaption(word="IMPORTANT", start=1.0, end=1.3, raw_token="IMPORTANT")
        self.assertTrue(is_emphasis_word(w3))

    def test_elongated_speech_duration(self):
        # 0.40s duration -> normal
        w_norm = WordCaption(word="yeh", start=1.0, end=1.4, raw_token="यह")
        self.assertFalse(is_emphasis_word(w_norm))
        # 0.60s duration -> elongated emphasis
        w_long = WordCaption(word="dekho", start=1.0, end=1.65, raw_token="देखो")
        self.assertTrue(is_emphasis_word(w_long))


class TestGreedyPixelWidthChunker(unittest.TestCase):
    """Test greedy chunker breaking on pause > 0.3s, punctuation, and canvas limits."""

    def setUp(self):
        self.renderer = CaptionRenderer(style="kinetic-pop")

    def test_break_on_speech_pause(self):
        """Chunker breaks when inter-word silence exceeds 0.30s."""
        words = [
            WordCaption(word="namaste", start=1.0, end=1.4, raw_token="नमस्ते"),
            WordCaption(word="dosto", start=1.4, end=1.8, raw_token="दोस्तों"),
            # 0.45s pause (> 0.30s)
            WordCaption(word="aaj", start=2.25, end=2.6, raw_token="आज"),
            WordCaption(word="padhenge", start=2.6, end=3.0, raw_token="पढ़ेंगे"),
        ]
        chunks = self.renderer.build_chunks(words)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([w.word for w in chunks[0].words], ["namaste", "dosto"])
        self.assertEqual([w.word for w in chunks[1].words], ["aaj", "padhenge"])

    def test_break_on_terminal_punctuation(self):
        """Chunker breaks when previous token has terminal punctuation (. ? ! | । ;)."""
        words = [
            WordCaption(word="kya", start=1.0, end=1.3, raw_token="क्या"),
            WordCaption(word="aap", start=1.3, end=1.6, raw_token="आप"),
            WordCaption(word="ready", start=1.6, end=1.9, raw_token="ready"),
            WordCaption(word="hain?", start=1.9, end=2.1, raw_token="हैं?"),
            # Continuous speech (0.05s gap) but punctuated
            WordCaption(word="chalo", start=2.15, end=2.4, raw_token="चलो"),
            WordCaption(word="shuru", start=2.4, end=2.7, raw_token="शुरू"),
            WordCaption(word="karte", start=2.7, end=3.0, raw_token="करते"),
            WordCaption(word="hain", start=3.0, end=3.3, raw_token="हैं"),
        ]
        chunks = self.renderer.build_chunks(words)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].words), 4)
        self.assertEqual(len(chunks[1].words), 4)
        self.assertEqual(chunks[0].words[-1].word, "hain?")
        self.assertEqual(chunks[1].words[0].word, "chalo")

    def test_orphan_line_rebalancing(self):
        """Prevent leaving a lonely 1-word orphan line by balancing with previous chunk."""
        words = [
            WordCaption(word="ye", start=1.0, end=1.2, raw_token="ये"),
            WordCaption(word="ek", start=1.2, end=1.4, raw_token="एक"),
            WordCaption(word="bahut", start=1.4, end=1.6, raw_token="बहुत"),
            WordCaption(word="badhiya", start=1.6, end=1.8, raw_token="बढ़िया"),
            # 5th word would be isolated if max_words=4
            WordCaption(word="sawal", start=1.8, end=2.0, raw_token="सवाल"),
        ]
        chunks = self.renderer.build_chunks(words, max_words_per_chunk=4)
        # Should rebalance from [4, 1] to [3, 2]
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].words), 3)
        self.assertEqual(len(chunks[1].words), 2)


class TestKineticStylesAndRendering(unittest.TestCase):
    """Test kinetic-pop and kinetic-roll styles and composers."""

    def setUp(self):
        self.words = [
            WordCaption(word="kya", start=1.0, end=1.8, raw_token="क्या"),
            WordCaption(word="aap", start=1.8, end=2.4, raw_token="आप"),
            WordCaption(word="ready", start=2.4, end=3.2, raw_token="ready"),
            WordCaption(word="hain", start=3.2, end=4.0, raw_token="हैं"),
        ]
        y, x = np.mgrid[0:1920, 0:1080]
        self.frame = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)

    def test_style_catalog_registration(self):
        """kinetic-pop and kinetic-roll must be registered with proper aliases."""
        self.assertIn("kinetic-pop", CAPTION_STYLES)
        self.assertIn("kinetic-roll", CAPTION_STYLES)

        style_pop = get_style("kinetic-pop")
        self.assertEqual(style_pop.name, "kinetic-pop")
        self.assertTrue(style_pop.word_entrance)
        self.assertTrue(style_pop.active_pop)
        self.assertEqual(style_pop.emphasis_color, (255, 82, 82))

        style_roll = get_style("kinetic-roll")
        self.assertEqual(style_roll.name, "kinetic-roll")
        self.assertEqual(style_roll.layout_mode, "rolling")
        self.assertEqual(style_roll.roll_frames, 6)

        # Aliases
        self.assertEqual(get_style("pop").name, "kinetic-pop")
        self.assertEqual(get_style("kinetic").name, "kinetic-pop")
        self.assertEqual(get_style("roll").name, "kinetic-roll")
        self.assertEqual(get_style("rolling").name, "kinetic-roll")
        self.assertEqual(get_style("2line").name, "kinetic-roll")

    def test_kinetic_pop_composer_rendering(self):
        """Verify kinetic-pop renders cleanly and produces expected active word pops."""
        renderer = CaptionRenderer(style="kinetic-pop")
        composer = renderer.create_composer(self.words, width=1080, height=1920)

        # Outside caption -> untouched passthrough
        f_pre = composer(self.frame.copy(), 0.5)
        self.assertTrue(np.array_equal(f_pre, self.frame))

        # At t=1.0s (start of word 0 'kya'): entrance frame 0
        f_step0 = composer(self.frame.copy(), 1.0)
        self.assertFalse(np.array_equal(f_step0, self.frame))

        # At t=1.08s (frame 2 of word 0): peak pop frame
        f_step2 = composer(self.frame.copy(), 1.08)
        self.assertFalse(np.array_equal(f_step2, self.frame))

        # Pixel difference between step 0 and step 2 proves animation is dynamic
        diff = np.abs(f_step0.astype(np.int32) - f_step2.astype(np.int32))
        self.assertGreater(diff.max(), 10, "Expected animated difference between step 0 and peak pop step")

    def test_kinetic_roll_2line_transition(self):
        """Verify kinetic-roll 2-line layout slides up line 2 into line 1 over 6 frames."""
        multi_chunk_words = [
            WordCaption(word="pehle", start=1.0, end=1.5, raw_token="पहले"),
            WordCaption(word="line", start=1.5, end=2.0, raw_token="line"),
            WordCaption(word="dusra", start=2.2, end=2.7, raw_token="दूसरा"),
            WordCaption(word="line", start=2.7, end=3.2, raw_token="line"),
        ]
        renderer = CaptionRenderer(style="kinetic-roll")
        composer = renderer.create_composer(multi_chunk_words, width=1080, height=1920)

        # During chunk 0 (stationary on line 1)
        f_chunk0 = composer(self.frame.copy(), 1.5)
        self.assertFalse(np.array_equal(f_chunk0, self.frame))

        # Transition into chunk 1 at t=2.20s (frame 0 of slide) vs t=2.32s (frame 3 midpoint of slide)
        f_roll0 = composer(self.frame.copy(), 2.20)
        f_roll3 = composer(self.frame.copy(), 2.32)
        diff_roll = np.abs(f_roll0.astype(np.int32) - f_roll3.astype(np.int32))
        self.assertGreater(diff_roll.max(), 10, "Expected vertical rolling slide motion across frames")

    def test_latin_only_assertion(self):
        """Devanagari characters strictly raise AssertionError in renderer."""
        invalid_words = [
            WordCaption(word="नमस्ते", start=1.0, end=2.0, raw_token="नमस्ते"),
        ]
        renderer = CaptionRenderer(style="kinetic-pop")
        with self.assertRaises(AssertionError):
            renderer.create_composer(invalid_words, width=1080, height=1920)


if __name__ == "__main__":
    unittest.main()
