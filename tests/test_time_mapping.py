#!/usr/bin/env python3
"""
test_time_mapping.py - Unit tests for time mapping and whisper cache slicing.

Verifies:
1. Time mapping formula: final_t = orig_t - clip_start + audio_delay_s
2. Word mapping with delay offsets.
3. Slice filtering and coverage verification.
4. Fallback triggers when coverage < 80%.
5. Disk cache save and load round-trip.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.whisper_backend import (
    Word,
    compute_lecture_cache_key,
    get_lecture_cache_path,
    load_lecture_transcript_cache,
    map_timestamp_to_clip,
    map_word_to_clip,
    save_lecture_transcript_cache,
    slice_cached_words_for_clip,
)


class TestTimeMapping(unittest.TestCase):
    """Test suite for timestamp mapping and cache slicing."""

    def test_formula_exact(self):
        """Verify final_t = orig_t - clip_start + audio_delay_s."""
        # Case A: Standard 500ms audio delay
        # At orig_t = 12.0s, clip starts at 10.0s -> offset is 2.0s + 0.5s = 2.5s
        t_final = map_timestamp_to_clip(orig_t=12.0, clip_start=10.0, audio_delay_ms=500.0)
        self.assertAlmostEqual(t_final, 2.5, places=3)

        # Case B: At clip start (orig_t == clip_start)
        # Audio is delayed by 500ms, so speech starts at 0.5s in the final clip
        t_start = map_timestamp_to_clip(orig_t=10.0, clip_start=10.0, audio_delay_ms=500.0)
        self.assertAlmostEqual(t_start, 0.5, places=3)

        # Case C: Zero audio delay
        t_zero = map_timestamp_to_clip(orig_t=25.0, clip_start=10.0, audio_delay_ms=0.0)
        self.assertAlmostEqual(t_zero, 15.0, places=3)

        # Case D: Custom delay 250ms
        t_custom = map_timestamp_to_clip(orig_t=15.0, clip_start=10.0, audio_delay_ms=250.0)
        self.assertAlmostEqual(t_custom, 5.25, places=3)

    def test_word_mapping(self):
        """Verify map_word_to_clip applies formula to start and end times."""
        orig_word = Word(start=14.2, end=14.8, text="Physics")
        mapped = map_word_to_clip(orig_word, clip_start=10.0, audio_delay_ms=500.0)

        # final_start = 14.2 - 10.0 + 0.5 = 4.7
        # final_end   = 14.8 - 10.0 + 0.5 = 5.3
        self.assertAlmostEqual(mapped.start, 4.7, places=3)
        self.assertAlmostEqual(mapped.end, 5.3, places=3)
        self.assertEqual(mapped.text, "Physics")
        self.assertEqual(mapped.word, "Physics")

    def test_slice_filtering(self):
        """Verify slice_cached_words_for_clip retains only words in the clip window."""
        cached_words = [
            Word(start=1.0, end=2.0, text="Intro"),        # Before clip [10.0, 20.0]
            Word(start=10.2, end=10.8, text="Concept"),    # In clip
            Word(start=15.0, end=15.5, text="Example"),    # In clip
            Word(start=19.2, end=19.8, text="Summary"),    # In clip
            Word(start=22.0, end=23.0, text="Outro"),      # After clip
        ]

        sliced = slice_cached_words_for_clip(
            cached_words=cached_words,
            clip_start=10.0,
            clip_end=20.0,
            audio_delay_ms=500.0,
            min_coverage=0.0,  # Bypass coverage check to test pure filtering
        )

        self.assertIsNotNone(sliced)
        self.assertEqual(len(sliced), 3)
        self.assertEqual([w.text for w in sliced], ["Concept", "Example", "Summary"])

        # Check mapped timestamps: orig_t - 10.0 + 0.5
        self.assertAlmostEqual(sliced[0].start, 0.7, places=2)  # 10.2 - 10.0 + 0.5
        self.assertAlmostEqual(sliced[1].start, 5.5, places=2)  # 15.0 - 10.0 + 0.5
        self.assertAlmostEqual(sliced[2].start, 9.7, places=2)  # 19.2 - 10.0 + 0.5

    def test_coverage_check_fallback(self):
        """Verify slice_cached_words_for_clip returns None when coverage < 80%."""
        # Simulated segment indicates 5 words of speech between 10.0s and 20.0s
        cached_segments = [
            {"id": 0, "start": 10.0, "end": 20.0, "text": "one two three four five"}
        ]

        # Case 1: Poor coverage (only 1 word covering 1 out of 5 words = 20% < 80%)
        sparse_words = [Word(start=12.0, end=12.5, text="sparse")]
        res_sparse = slice_cached_words_for_clip(
            cached_words=sparse_words,
            clip_start=10.0,
            clip_end=20.0,
            audio_delay_ms=500.0,
            cached_segments=cached_segments,
            min_coverage=0.80,
        )
        self.assertIsNone(res_sparse, "Should return None to trigger fallback on poor coverage (<80%)")

        # Case 2: Good coverage (5 out of 5 words = 100% >= 80%)
        dense_words = [
            Word(start=10.2, end=11.5, text="one"),
            Word(start=12.0, end=13.5, text="two"),
            Word(start=14.0, end=15.5, text="three"),
            Word(start=16.0, end=17.5, text="four"),
            Word(start=18.0, end=19.5, text="five"),
        ]
        res_dense = slice_cached_words_for_clip(
            cached_words=dense_words,
            clip_start=10.0,
            clip_end=20.0,
            audio_delay_ms=500.0,
            cached_segments=cached_segments,
            min_coverage=0.80,
        )
        self.assertIsNotNone(res_dense, "Should return sliced words when coverage >= 80%")
        self.assertEqual(len(res_dense), 5)

    def test_cache_save_and_load_roundtrip(self):
        """Verify save_lecture_transcript_cache and load_lecture_transcript_cache preserve words and segments."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_vid:
            tmp_path = Path(tmp_vid.name)
            model_name = "test-model-v1"

            sample_segments = [
                {"id": 0, "start": 0.0, "end": 4.5, "text": "Aaj hum padhenge physics."},
                {"id": 1, "start": 4.6, "end": 9.0, "text": "Ye bahut important concept hai."},
            ]
            sample_words = [
                Word(start=0.0, end=0.4, text="Aaj"),
                Word(start=0.4, end=0.7, text="hum"),
                Word(start=0.7, end=1.5, text="padhenge"),
                Word(start=1.5, end=2.2, text="physics"),
            ]

            saved_path = save_lecture_transcript_cache(
                video_path=tmp_path,
                model_name=model_name,
                segments=sample_segments,
                words=sample_words,
            )
            self.assertTrue(saved_path.exists())

            loaded = load_lecture_transcript_cache(tmp_path, model_name=model_name)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded["segments"]), 2)
            self.assertEqual(len(loaded["words"]), 4)
            self.assertEqual(loaded["words"][0].text, "Aaj")
            self.assertAlmostEqual(loaded["words"][0].start, 0.0, places=3)
            self.assertEqual(loaded["words"][3].text, "physics")
            self.assertAlmostEqual(loaded["words"][3].end, 2.2, places=3)

            # Cleanup test cache file
            saved_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
