#!/usr/bin/env python3
"""
test_health_check.py - Unit test for system health check and diagnostic suite.
Mocks external network calls to ensure fast, deterministic local test execution.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.health_check import run_diagnostics


class TestHealthCheck(unittest.TestCase):
    @patch("google.genai.Client")
    def test_run_diagnostics_mocked(self, mock_client_cls):
        """Verify health check executes and passes all local subsystem diagnostics."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock ping response
        mock_pong = MagicMock()
        mock_pong.text = "PONG"

        # Mock structured transliteration response
        mock_trans = MagicMock()
        mock_trans.text = '[{"i": 0, "r": "aaj"}, {"i": 1, "r": "ka"}, {"i": 2, "r": "event"}, {"i": 3, "r": "shandar"}, {"i": 4, "r": "tha"}]'

        # Mock structured hook response
        mock_hooks = MagicMock()
        mock_hooks.text = '[{"title": "Test Hook", "hook": "Test Hook Line", "start_segment_id": "[S0001]", "end_segment_id": "[S0002]", "start_time": 0.0, "end_time": 6.0, "duration": 6.0, "reason": "Engaging intro", "hook_strength": 9, "standalone": 9, "emotion": 8, "payoff": 8, "shareability": 9}]'

        mock_client.models.generate_content.side_effect = [mock_pong, mock_trans, mock_hooks]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "mock_test_key_1234567890"}):
            success = run_diagnostics(model_name="gemini-3.5-flash-lite")
            self.assertTrue(success, "Health check should succeed with mocked Gemini responses")


if __name__ == "__main__":
    unittest.main()
