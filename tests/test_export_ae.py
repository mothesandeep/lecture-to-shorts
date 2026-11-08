#!/usr/bin/env python3
"""
test_export_ae.py - Unit tests for After Effects ExtendScript JSX export and keyframe simplification.
"""

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np

from core.export_ae import (
    rdp_indices,
    simplify_keyframes,
    format_srt_time,
    format_timecode,
    generate_srt,
    generate_markers_csv,
    generate_extendscript_jsx,
    validate_jsx_syntax,
    export_ae_project,
)
from core.reframer import ReframePlan
from core.transcriber import WordCaption


class TestKeyframeSimplification(unittest.TestCase):
    """Test Ramer-Douglas-Peucker trajectory simplification."""

    def test_rdp_collinear_points(self):
        """Collinear points should be reduced to endpoints only."""
        t = np.linspace(0, 10, 50)
        x = 2.0 * t + 5.0
        y = -3.0 * t + 10.0
        pts = np.column_stack([x, y])

        idx = rdp_indices(pts, epsilon=0.1)
        self.assertEqual(idx, [0, 49])

    def test_rdp_preserves_extrema(self):
        """RDP must preserve peak and valley inflection points."""
        pts = np.array([
            [0.0, 0.0],
            [1.0, 10.0], # Peak
            [2.0, 0.0],
            [3.0, -10.0], # Valley
            [4.0, 0.0],
        ])
        idx = rdp_indices(pts, epsilon=1.0)
        self.assertIn(1, idx)
        self.assertIn(3, idx)

    def test_simplify_keyframes_rate(self):
        """Keyframe density must fall roughly within 5 to 10 keyframes per second."""
        fps = 25.0
        duration = 10.0
        n = int(fps * duration)
        t = np.linspace(0, duration, n)
        x = 540.0 + 80.0 * np.sin(t * 0.8)
        y = 960.0 + 30.0 * np.cos(t * 0.5)
        positions = np.column_stack([x, y])
        scales = 440.0 + 20.0 * np.sin(t * 0.4)

        kf_idx = simplify_keyframes(positions, scales, fps=fps, min_rate=5.0, max_rate=10.0)
        kf_rate = len(kf_idx) / duration

        self.assertGreaterEqual(kf_rate, 4.8)
        self.assertLessEqual(kf_rate, 10.5)
        self.assertEqual(kf_idx[0], 0)
        self.assertEqual(kf_idx[-1], n - 1)

    def test_simplify_keyframes_preserves_shot_cuts(self):
        """Shot cut frames must always be preserved across segments."""
        fps = 25.0
        n = 250
        positions = np.zeros((n, 2), dtype=np.float32)
        positions[:100, 0] = 300.0 # Shot 1
        positions[100:, 0] = 700.0 # Shot 2 (abrupt cut)
        scales = np.full(n, 400.0, dtype=np.float32)

        kf_idx = simplify_keyframes(positions, scales, fps=fps, cuts=[100])
        # Both frame 99 (end of shot 1) and frame 100 (start of shot 2) or cut point preserved
        self.assertTrue(99 in kf_idx or 100 in kf_idx)


class TestTimeAndSubtitleFormatters(unittest.TestCase):
    """Test SRT and CSV marker time formatting."""

    def test_format_srt_time(self):
        self.assertEqual(format_srt_time(0.0), "00:00:00,000")
        self.assertEqual(format_srt_time(1.234), "00:00:01,234")
        self.assertEqual(format_srt_time(65.5), "00:01:05,500")
        self.assertEqual(format_srt_time(3661.025), "01:01:01,025")

    def test_format_timecode(self):
        self.assertEqual(format_timecode(0.0, fps=25.0), "00:00:00:00")
        self.assertEqual(format_timecode(1.0, fps=25.0), "00:00:01:00")
        self.assertEqual(format_timecode(1.08, fps=25.0), "00:00:01:02") # Frame 2 at 25fps
        self.assertEqual(format_timecode(65.0, fps=25.0), "00:01:05:00")

    def test_generate_srt(self):
        words = [
            WordCaption(word="kya", start=1.0, end=1.5, raw_token="क्या"),
            WordCaption(word="aap", start=1.5, end=2.0, raw_token="आप"),
        ]
        with tempfile.TemporaryDirectory() as td:
            srt_path = Path(td) / "test.srt"
            generate_srt(words, srt_path)
            content = srt_path.read_text(encoding="utf-8")
            self.assertIn("1\n00:00:01,000 --> 00:00:01,500\nkya", content)
            self.assertIn("2\n00:00:01,500 --> 00:00:02,000\naap", content)

    def test_generate_markers_csv(self):
        words = [
            WordCaption(word="PW", start=0.5, end=1.0, raw_token="PW", emphasis=True),
            WordCaption(word="ROCKS", start=1.0, end=1.8, raw_token="ROCKS"),
        ]
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "markers.csv"
            generate_markers_csv(words, fps=25.0, out_path=csv_path)

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                self.assertEqual(reader[0][0], "Marker Name")
                self.assertEqual(reader[1][0], "PW")
                self.assertEqual(reader[2][0], "ROCKS")


class TestExtendScriptAndProjectExport(unittest.TestCase):
    """Test ExtendScript JSX generation, syntax validation, and full AE project export."""

    def test_jsx_has_no_json_parse(self):
        """ExtendScript JSX must embed literal DATA and strictly avoid JSON.parse."""
        dummy_data = {
            "compName": "TestComp",
            "compWidth": 1080,
            "compHeight": 1920,
            "fps": 25.0,
            "duration": 5.0,
            "footagePath": "/fake/snippet.mp4",
            "footageFilename": "snippet.mp4",
            "captionConfig": {"fontSize": 56, "bottomOffset": 380, "fontFamily": "Arial", "fillColor": [1, 1, 0], "strokeColor": [0, 0, 0], "strokeWidth": 4},
            "keyframes": {"times": [0.0, 0.2], "positions": [[540, 960, 0], [540, 960, 0]], "scales": [[400, 400, 100], [400, 400, 100]]},
            "words": [{"text": "TEST", "start": 1.0, "end": 2.0, "isEmphasis": False}],
        }
        with tempfile.TemporaryDirectory() as td:
            jsx_file = Path(td) / "test.jsx"
            generate_extendscript_jsx(dummy_data, jsx_file)
            content = jsx_file.read_text(encoding="utf-8")

            # Must contain native JS object assignment
            self.assertIn("var DATA = {", content)
            # Must NOT use JSON.parse
            self.assertNotIn("JSON.parse", content)
            # Must contain setValuesAtTimes
            self.assertIn("setValuesAtTimes", content)
            # Must contain addText
            self.assertIn("addText", content)

            # Node.js syntax validation
            valid, msg = validate_jsx_syntax(jsx_file)
            self.assertTrue(valid, f"Syntax validation failed: {msg}")

    def test_export_ae_project_position_scale_math(self):
        """Verify layer center anchor math: Position = (out_w/2 - (cx - W/2)*s, out_h/2 - (cy - H/2)*s)."""
        # Create a mock ReframePlan
        num_frames = 50
        fps = 25.0
        W, H = 854, 480
        out_w, out_h = 1080, 1920
        cw, ch = 246, 436

        # Center face at (cx=427, cy=240) -> matches source center
        crop_x0 = np.full(num_frames, (W - cw) / 2.0)
        crop_y0 = np.full(num_frames, (H - ch) / 2.0)

        plan = ReframePlan(
            src_path=Path("/fake/snippet.mp4"),
            out_path=Path("/fake/out.mp4"),
            width=W,
            height=H,
            fps=fps,
            total_frames=num_frames,
            num_frames=num_frames,
            crop_w=cw,
            crop_h=ch,
            out_canvas_w=out_w,
            out_canvas_h=out_h,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            scene_cuts=[0, num_frames],
            cut_indices=set(),
            layout="direct-crop",
            is_4_5=False,
            zoom=1.10,
            preset_name="talking_head",
            headroom=0.18,
            encoder_preset="fast",
            crf=18,
            audio_bitrate="192k",
        )

        words = [
            WordCaption(word="kya", start=0.5, end=1.0, raw_token="क्या"),
            WordCaption(word="aap", start=1.0, end=1.5, raw_token="आप"),
        ]

        with tempfile.TemporaryDirectory() as td:
            snip = Path(td) / "snippet_test.mp4"
            snip.write_bytes(b"dummy video content")

            out_export = Path(td) / "export_output"
            results = export_ae_project(
                snippet_path=snip,
                plan=plan,
                words=words,
                out_dir=out_export,
                comp_name="TestReel",
                target_rate=7.0,
                copy_snippet=False,
            )

            self.assertTrue(results["jsx"].is_file())
            self.assertTrue(results["srt"].is_file())
            self.assertTrue(results["csv"].is_file())

            # Read generated JSX and verify DATA values
            content = results["jsx"].read_text(encoding="utf-8")
            # When face is centered at (W/2, H/2), Position must be exactly (out_w/2, out_h/2) = (540, 960)
            self.assertIn("540.0", content)
            self.assertIn("960.0", content)
            # Scale must be (1920 / 436) * 100 = 440.37%
            expected_scale = round((1920.0 / 436.0) * 100.0, 2)
            self.assertIn(str(expected_scale), content)

            # Node syntax check
            valid, msg = validate_jsx_syntax(results["jsx"])
            self.assertTrue(valid, f"JSX validation failed: {msg}")


if __name__ == "__main__":
    unittest.main()
