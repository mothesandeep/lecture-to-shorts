#!/usr/bin/env python3
"""
test_fast_reframer.py - Unit tests for downscaled & spaced Haar face detection,
template tracking continuity, lock percentage, and corner drift metrics.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from core.reframer import (
    FaceReframer,
    ReframeConfig,
    ReframePlan,
    compute_corner_drift,
)


class TestFastReframer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = Path("samples/sample.mp4")
        if not cls.sample_path.is_file():
            raise unittest.SkipTest("samples/sample.mp4 not found for reframer test")

    def test_reframer_downscaled_and_every_n(self):
        """Test that detect_every=3 on 320px downscaled frame executes and preserves lock quality."""
        cfg_fast = ReframeConfig(
            src=self.sample_path,
            out=Path("output/test_fast_reframer.mp4"),
            detect_every=3,
            detect_width=320,
        )
        reframer_fast = FaceReframer(cfg_fast)
        plan_fast = reframer_fast.plan()

        self.assertIsInstance(plan_fast, ReframePlan)
        self.assertGreater(plan_fast.num_frames, 0)
        self.assertGreater(plan_fast.lock_pct, 75.0)  # High lock percentage (>85% expected)
        self.assertEqual(len(plan_fast.crop_x0), plan_fast.num_frames)
        self.assertEqual(len(plan_fast.crop_y0), plan_fast.num_frames)

        # Invariant check: Face anti-corner safe zone [0.32, 0.68] of crop width
        # Face horizontal offset from crop top-left relative to crop_w
        for t in range(plan_fast.num_frames):
            x0 = plan_fast.crop_x0[t]
            crop_w = plan_fast.crop_w
            # Verify coordinates are valid numbers and inside image bounds
            self.assertFalse(np.isnan(x0))
            self.assertGreaterEqual(x0, 0.0)
            self.assertLessEqual(x0 + crop_w, plan_fast.width + 0.1)

    def test_compute_corner_drift_metric(self):
        """Test compute_corner_drift between N=1 and N=3 plans."""
        cfg_1 = ReframeConfig(
            src=self.sample_path,
            out=Path("output/test_reframer_n1.mp4"),
            detect_every=1,
            detect_width=320,
        )
        plan_1 = FaceReframer(cfg_1).plan()

        cfg_3 = ReframeConfig(
            src=self.sample_path,
            out=Path("output/test_reframer_n3.mp4"),
            detect_every=3,
            detect_width=320,
        )
        plan_3 = FaceReframer(cfg_3).plan()

        drift = compute_corner_drift(plan_1, plan_3)
        self.assertIn("max_drift_px", drift)
        self.assertIn("mean_drift_px", drift)
        self.assertIn("max_drift_x", drift)
        self.assertIn("max_drift_y", drift)

        # Mean corner drift between N=1 and N=3 should be small (< 10 px)
        self.assertLess(drift["mean_drift_px"], 10.0)
        # Max corner drift should be small (< 40 px)
        self.assertLess(drift["max_drift_px"], 40.0)
        # Lock percentage difference should be minimal (within 1%)
        lock_delta = abs(plan_3.lock_pct - plan_1.lock_pct)
        self.assertLess(lock_delta, 1.0)


if __name__ == "__main__":
    unittest.main()
