#!/usr/bin/env python3
"""
test_punch_in.py - Unit tests for dynamic punch-in zoom, smoothstep easing,
face safe-zone invariant enforcement, and single fused warpAffine rendering.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from core.reframer import (
    detect_punch_in_events,
    enforce_centering_margin,
    enforce_headroom_margin,
    get_max_punch_in_zoom,
    smoothstep,
    zoom_curve,
)


class TestPunchInZoom(unittest.TestCase):
    """Test suite for punch-in zoom mathematics, event detection, and invariants."""

    def test_smoothstep_properties(self) -> None:
        """Verify smoothstep polynomial mathematical properties and boundary behavior."""
        # Boundaries
        self.assertEqual(smoothstep(0.0, 1.0, 0.0), 0.0)
        self.assertEqual(smoothstep(0.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(float(smoothstep(0.0, 1.0, 0.5)), 0.5, places=5)

        # Clamping outside edges
        self.assertEqual(smoothstep(1.0, 3.0, 0.0), 0.0)
        self.assertEqual(smoothstep(1.0, 3.0, 5.0), 1.0)

        # Vectorized support
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        s = smoothstep(0.0, 1.0, x)
        self.assertTrue(np.all(np.diff(s) > 0))  # Strictly monotonic

        # First derivative zero at boundaries: S'(u) = 6u(1 - u)
        eps = 1e-4
        deriv_0 = (smoothstep(0.0, 1.0, eps) - smoothstep(0.0, 1.0, 0.0)) / eps
        deriv_1 = (smoothstep(0.0, 1.0, 1.0) - smoothstep(0.0, 1.0, 1.0 - eps)) / eps
        self.assertLess(float(deriv_0), 0.01)
        self.assertLess(float(deriv_1), 0.01)

    def test_zoom_curve_generation(self) -> None:
        """Verify zoom curve bounds [1.0, z_max], smooth ramp in/out, and continuity."""
        fps = 25.0
        n = 100  # 4 seconds
        events = [(1.0, 2.5)]
        z_max = 1.25

        curve = zoom_curve(n, fps, events, z_max=z_max, ramp=0.35)
        self.assertEqual(len(curve), n)
        self.assertEqual(curve.dtype, np.float32)

        # All values strictly within [1.0, z_max]
        self.assertGreaterEqual(float(curve.min()), 1.0)
        self.assertLessEqual(float(curve.max()), z_max)

        # Outside events: base 1.0
        self.assertAlmostEqual(float(curve[0]), 1.0, places=4)
        self.assertAlmostEqual(float(curve[75]), 1.0, places=4)  # t = 3.0s

        # Peak hold reached at center of event (t = 1.75s, frame 43)
        self.assertAlmostEqual(float(curve[43]), z_max, places=4)

        # Continuity: no frame jumps (max frame-to-frame delta < 0.05)
        deltas = np.abs(np.diff(curve))
        self.assertLess(float(deltas.max()), 0.05, "Frame-to-frame zoom jump exceeded threshold")

    def test_zoom_curve_empty_events(self) -> None:
        """Verify zoom curve gracefully returns constant 1.0 when no events are provided."""
        curve = zoom_curve(50, 25.0, [], z_max=1.20)
        self.assertTrue(np.all(curve == 1.0))
        self.assertEqual(len(curve), 50)

    def test_get_max_punch_in_zoom_capping(self) -> None:
        """Verify resolution-dependent zoom capping (1.12 for <=480p, 1.25 for >=720p)."""
        self.assertEqual(get_max_punch_in_zoom(360), 1.12)
        self.assertEqual(get_max_punch_in_zoom(480), 1.12)
        self.assertEqual(get_max_punch_in_zoom(720), 1.25)
        self.assertEqual(get_max_punch_in_zoom(1080), 1.25)

        # Intermediate height linearly interpolated
        mid = get_max_punch_in_zoom(600)
        self.assertGreater(mid, 1.12)
        self.assertLess(mid, 1.25)

    def test_detect_punch_in_events(self) -> None:
        """Verify event detection from emphasis words and minimum gap enforcement."""
        words = [
            {"start": 1.0, "end": 1.3, "text": "normal"},
            {"start": 2.0, "end": 2.6, "text": "DHYAN SE SUNO!"},  # Exclamation mark
            {"start": 3.0, "end": 3.3, "text": "too close to previous"},
            {"start": 7.0, "end": 7.8, "text": "IMPORTANT"},       # Uppercase emphasis
            {"start": 12.0, "end": 12.6, "text": "SAMJHE?"},       # Question mark
        ]
        events = detect_punch_in_events(words=words, min_gap_s=3.0)
        self.assertGreaterEqual(len(events), 2)

        # Verify spacing: consecutive events must not start closer than min_gap_s * 0.5
        for i in range(len(events) - 1):
            gap = events[i + 1][0] - events[i][0]
            self.assertGreaterEqual(gap, 1.5, f"Events too close: {events[i]} and {events[i+1]}")

    def test_face_safe_zone_invariant_with_zoomed_crop_width(self) -> None:
        """
        Hard Invariant 5:
        Verify face center stays strictly within [0.32, 0.68] of the zoomed crop width.
        |crop_cx - face_cx| <= 0.18 * crop_w_arr[t].
        """
        n = 150
        # Simulating face walking across the frame
        face_cx = np.linspace(250.0, 600.0, n, dtype=np.float32)
        crop_cx = np.full(n, 427.0, dtype=np.float32)  # Centered on 854 width

        # Dynamic zoomed crop width: zooms in from 246 to 219 px and back
        z_curve = np.linspace(1.0, 1.25, n // 2)
        z_curve = np.concatenate([z_curve, np.linspace(1.25, 1.0, n - len(z_curve))])
        crop_w_arr = (270.0 / (1.10 * z_curve)).astype(np.float32)

        adjusted_cx = enforce_centering_margin(crop_cx, face_cx, crop_w_arr, max_deviation_ratio=0.18)

        # Check invariant on EVERY frame
        for i in range(n):
            w_t = float(crop_w_arr[i])
            f_x = float(face_cx[i])
            c_x = float(adjusted_cx[i])

            # Invariant: offset must be <= 0.18 * w_t
            offset_ratio = abs(f_x - c_x) / w_t
            self.assertLessEqual(
                offset_ratio,
                0.18001,
                f"Face safe-zone invariant breached at frame {i}: offset_ratio={offset_ratio:.4f} > 0.18"
            )

            # Left/right bounds of face within crop window:
            # Crop window is [c_x - w_t/2, c_x + w_t/2]
            face_pos_in_crop = (f_x - (c_x - w_t / 2.0)) / w_t
            self.assertGreaterEqual(face_pos_in_crop, 0.3199, f"Face too close to left corner: {face_pos_in_crop}")
            self.assertLessEqual(face_pos_in_crop, 0.6801, f"Face too close to right corner: {face_pos_in_crop}")

    def test_single_fused_warp_affine_direct_to_canvas(self) -> None:
        """
        Verify that single cv2.warpAffine with INTER_CUBIC correctly maps
        dynamic (x0, y0, w_crop, h_crop) directly to (canvas_w, canvas_h)
        with exact coordinate alignment and no black borders.
        """
        src_w, src_h = 854, 480
        canvas_w, canvas_h = 1080, 1920

        frame = np.ones((src_h, src_w, 3), dtype=np.uint8) * 128
        # Add colored markers in source frame
        cv2.rectangle(frame, (100, 50), (200, 300), (0, 255, 0), -1)

        # Simulated crop window during 1.20x punch-in zoom
        zoom_t = 1.20
        w_t = 270.0 / (1.10 * zoom_t)  # ~204.5 px
        h_t = 480.0 / (1.10 * zoom_t)  # ~363.6 px

        x0 = np.clip(350.0 - w_t / 2.0, 0.0, src_w - w_t)
        y0 = np.clip(180.0 - 0.18 * h_t, 0.0, src_h - h_t)

        sx = float(canvas_w) / w_t
        sy = float(canvas_h) / h_t

        M = np.float32([
            [sx, 0.0, -sx * x0],
            [0.0, sy, -sy * y0],
        ])

        canvas_frame = cv2.warpAffine(frame, M, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC)

        self.assertEqual(canvas_frame.shape, (canvas_h, canvas_w, 3))
        self.assertTrue(canvas_frame.flags.c_contiguous)

        # Verify affine transformation boundary points
        pt_tl = cv2.transform(np.array([[[x0, y0]]], dtype=np.float32), M)[0][0]
        pt_br = cv2.transform(np.array([[[x0 + w_t, y0 + h_t]]], dtype=np.float32), M)[0][0]

        np.testing.assert_allclose(pt_tl, [0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(pt_br, [canvas_w, canvas_h], atol=1e-3)

        # Verify no uninitialized black borders in destination canvas
        self.assertGreater(float(canvas_frame.mean()), 100.0)


if __name__ == "__main__":
    unittest.main()
