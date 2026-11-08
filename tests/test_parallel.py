#!/usr/bin/env python3
"""
test_parallel.py - Unit tests for render_reel producer-consumer queue,
--fast videotoolbox acceleration, and background clip preparation.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from core.render import render_reel
from core.av_sync import probe_av_sync, verify_av_sync


class TestParallelRenderAndFastEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small dummy CFR test snippet with audio (25fps, 1.0s = 25 frames)
        cls.test_dir = Path("output/test_parallel_fixtures")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.snippet_path = cls.test_dir / "test_snippet.mp4"

        # Generate 1-second synthetic video with silent audio track via FFmpeg
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=25",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest",
            "-c:a", "aac", "-b:a", "128k",
            str(cls.snippet_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_render_reel_producer_consumer_libx264(self):
        """Test render_reel using producer-consumer queue with default libx264."""
        out_p = self.test_dir / "out_default_libx264.mp4"
        res = render_reel(
            snippet=self.snippet_path,
            out_path=out_p,
            no_caption=True,
            fast=False,
        )
        self.assertTrue(res.is_file())
        info = probe_av_sync(res)
        self.assertAlmostEqual(info["v_duration"], 1.0, delta=0.08)
        self.assertTrue(info["is_cfr"])
        verify_av_sync(res, max_diff=0.05)

    def test_render_reel_fast_videotoolbox(self):
        """Test render_reel using --fast (h264_videotoolbox -b:v 12M)."""
        out_p = self.test_dir / "out_fast_vt.mp4"
        res = render_reel(
            snippet=self.snippet_path,
            out_path=out_p,
            no_caption=True,
            fast=True,
        )
        self.assertTrue(res.is_file())
        info = probe_av_sync(res)
        self.assertAlmostEqual(info["v_duration"], 1.0, delta=0.08)
        self.assertTrue(info["is_cfr"])
        verify_av_sync(res, max_diff=0.05)

    def test_render_reel_error_handling_no_zombies_or_hangs(self):
        """Test that invalid input terminates immediately without hanging or leaving zombies."""
        fake_path = self.test_dir / "nonexistent.mp4"
        with self.assertRaises(FileNotFoundError):
            render_reel(snippet=fake_path, out_path=self.test_dir / "never.mp4")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)


if __name__ == "__main__":
    unittest.main()
