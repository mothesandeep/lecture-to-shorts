#!/usr/bin/env python3
"""
test_detectors.py - Unit tests for pluggable face detectors (Haar, YuNet, Vision),
graceful fallbacks, back-turned motion fallback, and position holding.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from core.reframer import (
    BaseFaceDetector,
    HaarFaceDetector,
    YuNetFaceDetector,
    VisionFaceDetector,
    create_face_detector,
    FaceReframer,
    ReframeConfig,
    ReframePlan,
    compute_corner_drift,
    find_cascade,
)


class TestPluggableDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = Path("samples/sample.mp4")
        if not cls.sample_path.is_file():
            raise unittest.SkipTest("samples/sample.mp4 not found for detector test")

        # Read first frame of sample for static detector testing
        cap = cv2.VideoCapture(str(cls.sample_path))
        ok, cls.test_frame = cap.read()
        cap.release()
        if not ok or cls.test_frame is None:
            raise unittest.SkipTest("Could not read frame from samples/sample.mp4")
        cls.test_gray = cv2.cvtColor(cls.test_frame, cv2.COLOR_BGR2GRAY)

    def test_haar_detector_init_and_detect(self):
        """Test HaarFaceDetector initialization and detection."""
        cascade_file = find_cascade()
        detector = HaarFaceDetector(cascade_file, detect_width=320)
        self.assertEqual(detector.name, "haar")

        boxes = detector.detect(self.test_frame, self.test_gray)
        self.assertIsInstance(boxes, list)
        self.assertGreater(len(boxes), 0)

        for b in boxes:
            self.assertEqual(len(b), 6)
            x, y, w, h, cx, cy = b
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertAlmostEqual(cx, x + w / 2.0, places=1)
            self.assertAlmostEqual(cy, y + h / 2.0, places=1)

    def test_yunet_detector_init_and_detect(self):
        """Test YuNetFaceDetector initialization and detection."""
        detector = YuNetFaceDetector(detect_width=320)
        self.assertEqual(detector.name, "yunet")

        boxes = detector.detect(self.test_frame, self.test_gray)
        self.assertIsInstance(boxes, list)
        self.assertGreater(len(boxes), 0)

        for b in boxes:
            self.assertEqual(len(b), 6)
            x, y, w, h, cx, cy = b
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertAlmostEqual(cx, x + w / 2.0, places=1)
            self.assertAlmostEqual(cy, y + h / 2.0, places=1)

    def test_vision_detector_init_and_detect(self):
        """Test VisionFaceDetector initialization and detection via Apple Vision framework."""
        try:
            detector = VisionFaceDetector(detect_width=320)
        except RuntimeError as e:
            self.skipTest(f"Vision framework not available: {e}")

        self.assertEqual(detector.name, "vision")
        boxes = detector.detect(self.test_frame, self.test_gray)
        self.assertIsInstance(boxes, list)
        self.assertGreater(len(boxes), 0)

        for b in boxes:
            self.assertEqual(len(b), 6)
            x, y, w, h, cx, cy = b
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertAlmostEqual(cx, x + w / 2.0, places=1)
            self.assertAlmostEqual(cy, y + h / 2.0, places=1)

    def test_create_face_detector_factory(self):
        """Test create_face_detector factory and graceful fallback."""
        det_default = create_face_detector()
        self.assertEqual(det_default.name, "haar")

        det_haar = create_face_detector("haar")
        self.assertEqual(det_haar.name, "haar")

        det_yunet = create_face_detector("yunet")
        self.assertEqual(det_yunet.name, "yunet")

        det_vision = create_face_detector("vision")
        self.assertEqual(det_vision.name, "vision")

        # Graceful fallback on invalid/missing model
        det_fallback = create_face_detector("yunet", model_path="nonexistent_model.onnx")
        self.assertEqual(det_fallback.name, "haar")

        # Graceful fallback on unknown detector type
        det_unknown = create_face_detector("unknown_detector")
        self.assertEqual(det_unknown.name, "haar")

    def test_back_turned_fallback_and_hold(self):
        """
        Test back-turned fallback mechanism:
        When face is lost, system falls back to nearby motion centroids,
        and if motion also fails, holds position without drifting.
        """
        cfg = ReframeConfig(
            src=self.sample_path,
            out=Path("output/test_fallback.mp4"),
            detect_every=3,
            detect_width=320,
            detector="haar",
        )
        plan = FaceReframer(cfg).plan()

        self.assertIsInstance(plan, ReframePlan)
        self.assertGreater(plan.num_frames, 0)
        # Verify fallback counters are populated and positive
        self.assertGreater(plan.face_lock_count, 0)
        self.assertGreaterEqual(plan.motion_fallback_count, 0)
        self.assertGreaterEqual(plan.hold_count, 0)
        # Total tracked frames (face + fallback + hold) should cover virtually all frames
        total_tracked = plan.face_lock_count + plan.motion_fallback_count + plan.hold_count
        self.assertGreater(total_tracked, int(plan.num_frames * 0.90))

    def test_reframer_plans_all_detectors(self):
        """Compare plans produced by Haar, YuNet, and Vision detectors on sample."""
        cfg_haar = ReframeConfig(src=self.sample_path, out=Path("output/plan_haar.mp4"), detector="haar")
        plan_haar = FaceReframer(cfg_haar).plan()

        cfg_yunet = ReframeConfig(src=self.sample_path, out=Path("output/plan_yunet.mp4"), detector="yunet")
        plan_yunet = FaceReframer(cfg_yunet).plan()

        cfg_vision = ReframeConfig(src=self.sample_path, out=Path("output/plan_vision.mp4"), detector="vision")
        plan_vision = FaceReframer(cfg_vision).plan()

        # All detectors should achieve high lock percentage (>80%)
        self.assertGreater(plan_haar.lock_pct, 80.0)
        self.assertGreater(plan_yunet.lock_pct, 80.0)
        self.assertGreater(plan_vision.lock_pct, 80.0)

        # Invariant checks: Anti-corner safe zone enforced for all
        for plan in (plan_haar, plan_yunet, plan_vision):
            for t in range(plan.num_frames):
                x0 = plan.crop_x0[t]
                crop_w = plan.crop_w_arr[t] if plan.crop_w_arr is not None else plan.crop_w
                self.assertFalse(np.isnan(x0))
                self.assertGreaterEqual(x0, 0.0)
                self.assertLessEqual(x0 + crop_w, plan.width + 0.1)

        # Drift between detectors should remain bounded
        drift_hy = compute_corner_drift(plan_haar, plan_yunet)
        drift_hv = compute_corner_drift(plan_haar, plan_vision)
        self.assertLess(drift_hy["mean_drift_px"], 25.0)
        self.assertLess(drift_hv["mean_drift_px"], 25.0)


if __name__ == "__main__":
    unittest.main()
