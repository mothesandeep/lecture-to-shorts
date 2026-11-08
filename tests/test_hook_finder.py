#!/usr/bin/env python3
"""
test_hook_finder.py - Comprehensive unit tests for hook_finder.py and audio_features.py.
Mocks all Gemini API calls so NO real network requests are executed.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from core.audio_features import AudioFeatureExtractor
from core.hook_finder import (
    DEFAULT_VIRAL_SCORE_WEIGHTS,
    HookFinder,
    ViralClipCandidate,
    ViralClipMetadata,
    check_face_presence,
    compute_weighted_viral_score,
    deduplicate_candidate_clips,
    normalize_segment_id,
)
from core.whisper_backend import Word


class TestAudioFeatures(unittest.TestCase):
    def test_audio_feature_extractor_synthetic(self):
        """Test energy z-scores, speech rate, and pause density with synthetic data."""
        sr = 16000
        # 10s audio: first 5s loud, next 5s soft
        t = np.linspace(0, 10, 10 * sr, endpoint=False)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        samples[5 * sr:] *= 0.1

        extractor = AudioFeatureExtractor(samples=samples, sample_rate=sr)
        self.assertGreater(len(extractor.energy_zscores), 0)

        z_loud = extractor.get_energy_zscore(0.0, 4.0)
        z_soft = extractor.get_energy_zscore(6.0, 10.0)
        self.assertGreater(z_loud, z_soft)

        # Speech rate & pause density
        words = [
            Word(start=1.0, end=1.2, text="Pehle"),
            Word(start=1.3, end=1.5, text="baat"),
            Word(start=1.6, end=1.8, text="samjho"),
            Word(start=2.5, end=2.8, text="zaruri"),  # 0.7s pause before this word
        ]
        rate = extractor.compute_speech_rate(words, 1.0, 3.0)
        self.assertAlmostEqual(rate, 2.0, delta=0.1)

        pause_density = extractor.compute_pause_density(words, 1.0, 3.0, pause_threshold_s=0.35)
        self.assertGreater(pause_density, 0.20)

        annotated = extractor.format_annotated_segment(
            seg_id_str="[S0001]",
            start_t=1.0,
            end_t=3.0,
            text="Pehle baat samjho zaruri",
            words=words
        )
        self.assertTrue(annotated.startswith("[S0001]"))
        self.assertIn("energy", annotated)
        self.assertIn("Pehle baat samjho zaruri", annotated)


class TestHookFinderUnit(unittest.TestCase):
    def test_segment_id_normalization(self):
        """Test normalization of various segment id representations."""
        self.assertEqual(normalize_segment_id("S0001"), "S0001")
        self.assertEqual(normalize_segment_id("[S0005]"), "S0005")
        self.assertEqual(normalize_segment_id("S12"), "S0012")
        self.assertEqual(normalize_segment_id("42"), "S0042")
        self.assertEqual(normalize_segment_id(7), "S0007")
        self.assertIsNone(normalize_segment_id(None))

    def test_weighted_viral_score_computation(self):
        """Test calculation of composite score from sub-scores."""
        sub_scores_perfect = {
            "hook_strength": 10.0,
            "standalone": 10.0,
            "emotion": 10.0,
            "payoff": 10.0,
            "shareability": 10.0,
        }
        score_100 = compute_weighted_viral_score(sub_scores_perfect)
        self.assertEqual(score_100, 100)

        sub_scores_moderate = {
            "hook_strength": 8.0,
            "standalone": 7.0,
            "emotion": 8.0,
            "payoff": 7.0,
            "shareability": 8.0,
        }
        score_mod = compute_weighted_viral_score(sub_scores_moderate)
        self.assertGreaterEqual(score_mod, 70)
        self.assertLessEqual(score_mod, 85)

        # Scale 1-100 normalization
        sub_scores_100_scale = {k: v * 10 for k, v in sub_scores_moderate.items()}
        score_normalized = compute_weighted_viral_score(sub_scores_100_scale)
        self.assertEqual(score_normalized, score_mod)

    def test_candidate_deduplication(self):
        """Test deduplicating overlapping candidate clips from adjacent map windows."""
        clip_a = ViralClipMetadata(
            clip_id=1,
            title="Clip A Higher Score",
            hook="Hook A",
            start_time=10.0,
            end_time=40.0,
            duration=30.0,
            viral_score=92,
            reason="High viral potential",
            slug="clip_a",
            aspect_ratio="9:16",
        )
        # Clip B overlaps with Clip A by 25 seconds (> 80%) but has lower score
        clip_b = ViralClipMetadata(
            clip_id=2,
            title="Clip B Duplicate",
            hook="Hook B",
            start_time=15.0,
            end_time=45.0,
            duration=30.0,
            viral_score=80,
            reason="Duplicate",
            slug="clip_b",
            aspect_ratio="9:16",
        )
        # Clip C is distinct (no overlap)
        clip_c = ViralClipMetadata(
            clip_id=3,
            title="Clip C Distinct",
            hook="Hook C",
            start_time=90.0,
            end_time=120.0,
            duration=30.0,
            viral_score=85,
            reason="Standalone moment",
            slug="clip_c",
            aspect_ratio="9:16",
        )

        deduped = deduplicate_candidate_clips([clip_a, clip_b, clip_c])
        self.assertEqual(len(deduped), 2)
        slugs = [c.slug for c in deduped]
        self.assertIn("clip_a", slugs)
        self.assertIn("clip_c", slugs)
        self.assertNotIn("clip_b", slugs)

    def test_convert_candidate_to_metadata(self):
        """Test converting ViralClipCandidate with segment IDs to ViralClipMetadata with seconds."""
        finder = HookFinder(api_key="mock_key")
        segment_map = {
            "S0001": {"seg_id": "S0001", "start": 0.0, "end": 4.5, "text": "Start here"},
            "S0002": {"seg_id": "S0002", "start": 4.5, "end": 15.0, "text": "Middle part"},
            "S0003": {"seg_id": "S0003", "start": 15.0, "end": 32.0, "text": "Resolution here"},
        }
        cand = ViralClipCandidate(
            start_segment_id="S0001",
            end_segment_id="S0003",
            title="Sapna Sach Hoga",
            hook="Kitne logon ne socha tha?",
            aspect_ratio="9:16",
            hook_strength=9,
            standalone=8,
            emotion=9,
            payoff=8,
            shareability=9,
            reason="Emotionally charged opening",
            slug="sapna_sach_hoga",
        )
        meta = finder._convert_candidate_to_metadata(cand, segment_map, total_duration=32.0)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.start_time, 0.0)
        self.assertEqual(meta.end_time, 32.0)
        self.assertEqual(meta.duration, 32.0)
        self.assertGreaterEqual(meta.viral_score, 85)
        self.assertEqual(meta.start_segment_id, "S0001")
        self.assertEqual(meta.end_segment_id, "S0003")

    @patch("core.hook_finder.AudioFeatureExtractor.from_file")
    @patch("google.genai.Client")
    def test_mocked_gemini_find_viral_clips(self, mock_genai_client_class, mock_audio_extractor_from_file):
        """Test full find_viral_clips pipeline using mocked Gemini client (no real API calls)."""
        # Mock audio extractor
        mock_extractor = AudioFeatureExtractor()
        mock_audio_extractor_from_file.return_value = mock_extractor

        # Mock Gemini client
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Mock candidates returned by Gemini
        mock_candidate_json = json.dumps([
            {
                "start_segment_id": "S0001",
                "end_segment_id": "S0002",
                "title": "Stage Share Karne Ka Sapna",
                "hook": "Imandari se batana kitne logon ne wo moment jiya?",
                "aspect_ratio": "9:16",
                "hook_strength": 9,
                "standalone": 9,
                "emotion": 9,
                "payoff": 8,
                "shareability": 9,
                "reason": "Deep emotional resonance",
                "slug": "stage_share_sapna",
            },
            {
                "start_segment_id": "S0003",
                "end_segment_id": "S0004",
                "title": "Conditions Kabhi Nahi Sudrengi",
                "hook": "Agar aap intezaar karte rahoge to kabhi nahi hoga",
                "aspect_ratio": "9:16",
                "hook_strength": 9,
                "standalone": 8,
                "emotion": 9,
                "payoff": 9,
                "shareability": 9,
                "reason": "Motivational reality check",
                "slug": "conditions_nahi_sudrengi",
            }
        ])

        mock_response = MagicMock()
        mock_response.text = mock_candidate_json
        mock_client.models.generate_content.return_value = mock_response

        finder = HookFinder(api_key="mock_key_for_test", min_duration=10.0, max_duration=40.0)

        # Mock transcript returning 4 segments
        fake_segments = [
            {"id": 1, "seg_id": "S0001", "start": 0.0, "end": 8.0, "text": "Imandari se batana kitne logon ne"},
            {"id": 2, "seg_id": "S0002", "start": 8.0, "end": 20.0, "text": "wo moment jiya tha stage share karne ka"},
            {"id": 3, "seg_id": "S0003", "start": 25.0, "end": 35.0, "text": "Agar aap intezaar karte rahoge"},
            {"id": 4, "seg_id": "S0004", "start": 35.0, "end": 50.0, "text": "to kabhi nahi badlega yeh"},
        ]
        finder.get_full_transcript = MagicMock(return_value=fake_segments)

        clips = finder.find_viral_clips(Path("dummy_video.mp4"), num_clips=2)

        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0].title, "Stage Share Karne Ka Sapna")
        self.assertEqual(clips[0].start_time, 0.0)
        self.assertEqual(clips[0].end_time, 20.0)
        self.assertGreaterEqual(clips[0].viral_score, 80)
        self.assertEqual(clips[1].title, "Conditions Kabhi Nahi Sudrengi")
        self.assertEqual(clips[1].start_time, 25.0)
        self.assertEqual(clips[1].end_time, 50.0)

    @patch("core.hook_finder.check_face_presence")
    @patch("core.hook_finder.AudioFeatureExtractor.from_file")
    @patch("google.genai.Client")
    def test_face_presence_filter(self, mock_genai_client_class, mock_audio_extractor, mock_check_face):
        """Test optional face presence filter dropping 9:16 clips where teacher is visible < 70%."""
        mock_audio_extractor.return_value = AudioFeatureExtractor()
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        mock_candidate_json = json.dumps([
            {
                "start_segment_id": "S0001",
                "end_segment_id": "S0002",
                "title": "Low Face Presence Clip",
                "hook": "Look at the slide",
                "aspect_ratio": "9:16",
                "hook_strength": 8,
                "standalone": 8,
                "emotion": 7,
                "payoff": 7,
                "shareability": 7,
                "reason": "Slide presentation",
                "slug": "low_face",
            },
            {
                "start_segment_id": "S0003",
                "end_segment_id": "S0004",
                "title": "High Face Presence Clip",
                "hook": "Teacher speaking face to face",
                "aspect_ratio": "9:16",
                "hook_strength": 9,
                "standalone": 9,
                "emotion": 9,
                "payoff": 9,
                "shareability": 9,
                "reason": "Strong connection",
                "slug": "high_face",
            }
        ])
        mock_response = MagicMock()
        mock_response.text = mock_candidate_json
        mock_client.models.generate_content.return_value = mock_response

        # Candidates are processed in descending score order (High Face has score 90, Low Face has score 73)
        # High Face gets (True, 0.90) and passes; Low Face gets (False, 0.50) and is dropped.
        mock_check_face.side_effect = [
            (True, 0.90),
            (False, 0.50),
        ]

        finder = HookFinder(
            api_key="mock_key",
            min_duration=10.0,
            max_duration=40.0,
            filter_face_presence=True
        )
        fake_segments = [
            {"id": 1, "seg_id": "S0001", "start": 0.0, "end": 10.0, "text": "Slide text"},
            {"id": 2, "seg_id": "S0002", "start": 10.0, "end": 20.0, "text": "More slide"},
            {"id": 3, "seg_id": "S0003", "start": 30.0, "end": 40.0, "text": "Teacher talking"},
            {"id": 4, "seg_id": "S0004", "start": 40.0, "end": 55.0, "text": "Conclusion"},
        ]
        finder.get_full_transcript = MagicMock(return_value=fake_segments)

        # Mock video file exists
        with patch.object(Path, "is_file", return_value=True):
            clips = finder.find_viral_clips(Path("dummy_video.mp4"), num_clips=2)

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].title, "High Face Presence Clip")

    @patch("core.hook_finder.AudioFeatureExtractor.from_file")
    @patch("google.genai.Client")
    def test_map_reduce_long_lecture(self, mock_genai_client_class, mock_audio_extractor):
        """Test Map-Reduce workflow triggering on lectures > map_reduce_threshold (600s)."""
        mock_audio_extractor.return_value = AudioFeatureExtractor()
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Window 1 response
        win1_json = json.dumps([
            {
                "start_segment_id": "S0001",
                "end_segment_id": "S0002",
                "title": "Early Breakthrough",
                "hook": "Pehle 5 minute mein kya hua?",
                "aspect_ratio": "9:16",
                "hook_strength": 9,
                "standalone": 8,
                "emotion": 8,
                "payoff": 8,
                "shareability": 8,
                "reason": "Early momentum",
                "slug": "early_breakthrough",
            }
        ])
        # Window 2 response
        win2_json = json.dumps([
            {
                "start_segment_id": "S0003",
                "end_segment_id": "S0004",
                "title": "Late Climax",
                "hook": "Aakhiri waqt pe haar mat maano",
                "aspect_ratio": "9:16",
                "hook_strength": 9,
                "standalone": 9,
                "emotion": 10,
                "payoff": 9,
                "shareability": 10,
                "reason": "Powerful climax",
                "slug": "late_climax",
            }
        ])

        # Rerank selection response
        rerank_json = json.dumps({"selected_slugs": ["late_climax", "early_breakthrough"]})

        mock_resp1 = MagicMock()
        mock_resp1.text = win1_json
        mock_resp2 = MagicMock()
        mock_resp2.text = win2_json
        mock_resp_rerank = MagicMock()
        mock_resp_rerank.text = rerank_json

        mock_client.models.generate_content.side_effect = [
            mock_resp1,
            mock_resp2,
            mock_resp_rerank,
        ]

        finder = HookFinder(
            api_key="mock_key",
            min_duration=20.0,
            max_duration=60.0,
            map_reduce_threshold=500.0  # Trigger threshold at 500s
        )

        # 800s total duration
        fake_segments = [
            {"id": 1, "seg_id": "S0001", "start": 10.0, "end": 35.0, "text": "Pehle 5 minute mein kya hua?"},
            {"id": 2, "seg_id": "S0002", "start": 35.0, "end": 50.0, "text": "wo dekhna zaruri tha."},
            {"id": 3, "seg_id": "S0003", "start": 620.0, "end": 645.0, "text": "Aakhiri waqt pe haar mat maano."},
            {"id": 4, "seg_id": "S0004", "start": 645.0, "end": 800.0, "text": "Kyunki wahi se jeet shuru hoti hai."},
        ]
        finder.get_full_transcript = MagicMock(return_value=fake_segments)

        clips = finder.find_viral_clips(Path("long_lecture.mp4"), num_clips=2)
        self.assertEqual(len(clips), 2)
        # Rerank ordered late_climax first
        self.assertEqual(clips[0].title, "Late Climax")
        self.assertEqual(clips[1].title, "Early Breakthrough")


if __name__ == "__main__":
    unittest.main()
