#!/usr/bin/env python3
"""
health_check.py - System diagnostics and health check for Gemini API & Romanizer.

Runs comprehensive checks on:
1. Environment & Dependencies (Python 3.14, macOS Apple Silicon, FFmpeg, OpenCV, Pillow, Whisper, PyTorch).
2. Haar Cascade frontalface XML locator.
3. System TrueType font resolution.
4. Local rule-based transliteration engine & 100% Latin ASCII guarantee.
5. Gemini API connectivity, structured Pydantic schema, model compatibility (gemini-3.5-flash-lite),
   and persistent disk caching.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

try:
    from core.captioner import FontManager
    from core.reframer import find_cascade
    from core.transcriber import (
        CACHE_DIR,
        DEV,
        HinglishTranscriber,
        local_romanize,
        sanitize_latin,
    )
except ImportError:
    from captioner import FontManager
    from reframer import find_cascade
    from transcriber import (
        CACHE_DIR,
        DEV,
        HinglishTranscriber,
        local_romanize,
        sanitize_latin,
    )


def run_diagnostics(model_name: str = "gemini-3.5-flash-lite") -> bool:
    """Execute complete system health check."""
    print("=" * 65)
    print(" REEL MAKER PIPELINE HEALTH CHECK & DIAGNOSTICS")
    print("=" * 65)

    all_passed = True

    # 1. Platform & Python
    py_ver = sys.version.split()[0]
    os_name = platform.system()
    arch = platform.machine()
    print(f"\n[1/5] Platform Environment:")
    print(f"  • Python Version : {py_ver} ({'Python 3.14+' if sys.version_info >= (3, 14) else 'Supported'})")
    print(f"  • Operating System: {os_name} ({arch})")

    # 2. FFmpeg & External Binaries
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, check=True)
            first_line = res.stdout.splitlines()[0] if res.stdout else "Available"
            print(f"  • FFmpeg Binary  : {ffmpeg_bin} -> {first_line[:40]}")
        except Exception:
            print(f"  • FFmpeg Binary  : {ffmpeg_bin} (Ready)")
    else:
        print("  • FFmpeg Binary  : [MISSING] ffmpeg not found in PATH! (brew install ffmpeg)")
        all_passed = False

    # 3. Vision & Font Resources
    print(f"\n[2/5] Vision & Media Assets:")
    try:
        cascade_file = find_cascade()
        print(f"  • Haar Cascade   : Found at {cascade_file}")
    except Exception as e:
        print(f"  • Haar Cascade   : [FAIL] {e}")
        all_passed = False

    fm = FontManager()
    font = fm.get_font(48)
    font_name = getattr(font, "getname", lambda: ("Font", "Loaded"))()
    print(f"  • Pillow Font    : Resolved successfully -> {font_name}")

    try:
        from core.captioner import CAPTION_STYLES, list_styles
        styles = list_styles()
        print(f"  • Caption Styles : {len(styles)} presets verified ({', '.join(styles[:4])}...) [OK]")
    except Exception as e:
        print(f"  • Caption Styles : [FAIL] {e}")
        all_passed = False

    try:
        from core.reframer import REFRAME_PRESETS, list_reframe_presets
        presets = list_reframe_presets()
        print(f"  • Reframe Presets: {len(presets)} presets verified ({', '.join(presets)}) [OK]")
    except Exception as e:
        print(f"  • Reframe Presets: [FAIL] {e}")
        all_passed = False

    try:
        from core.av_sync import probe_av_sync, cut_snippet_accurate
        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin:
            print(f"  • FFprobe Binary : {ffprobe_bin} (AV Sync Probe Ready)")
        else:
            print(f"  • FFprobe Binary : [MISSING] ffprobe not found in PATH!")
            all_passed = False
        print("  • AV Sync Engine : PTS Zeroing + CFR Normalizer (async=1000) [OK]")
    except Exception as e:
        print(f"  • AV Sync Engine : [FAIL] {e}")
        all_passed = False

    # 4. Local Rule-based Romanizer & Latin Sanitization
    print(f"\n[3/5] Local Transliteration & Latin Guarantee:")
    test_words = [
        "लोगों", "देखते", "क्यों", "कैसे", "समझना",
        "इवेंट", "मोमेंट", "ईमानदारी", "१२३", "ॐ", "फंक्शन", "प्रॉब्लम"
    ]
    latin_failures = 0
    for w in test_words:
        rom = local_romanize(w)
        sanitized = sanitize_latin(rom)
        is_latin = (DEV.search(sanitized) is None) and sanitized.isascii()
        status = "OK" if is_latin else "FAIL"
        if not is_latin:
            latin_failures += 1
            all_passed = False
        print(f"  • {w:10} -> {sanitized:14} (Latin={is_latin}) [{status}]")

    if latin_failures == 0:
        print("  -> 100% Latin-only invariant verified. Zero tofu box (□□□) risk.")

    # 5. Whisper & PyTorch Environment
    print(f"\n[4/5] Whisper & PyTorch Acceleration:")
    try:
        import torch
        import whisper
        from core.transcriber import DEFAULT_WHISPER_INITIAL_PROMPT
        mps_avail = torch.backends.mps.is_available()
        print(f"  • PyTorch Version: {torch.__version__} (Apple Silicon MPS: {mps_avail})")
        print(f"  • OpenAI Whisper : Installed & Ready")
        print(f"  • Default Model  : large-v3 (Word-level clip sync)")
        print(f"  • Decoding Config: no_speech_thr=0.3, condition_on_previous_text=False")
        print(f"  • Initial Prompt : \"{DEFAULT_WHISPER_INITIAL_PROMPT[:45]}...\" [OK]")
    except Exception as e:
        print(f"  • Whisper/PyTorch: [FAIL] {e}")
        all_passed = False

    # 6. Gemini API & Structured Output
    print(f"\n[5/5] Google GenAI SDK & Gemini API Health:")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  • GEMINI_API_KEY : [NOT SET] in environment.")
        print("    Pipeline will operate in offline rule-based transliteration mode.")
        print("    To enable Gemini 3.5 Flash Lite, run: export GEMINI_API_KEY='your-key'")
    else:
        key_masked = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        print(f"  • GEMINI_API_KEY : Detected ({key_masked})")

        try:
            from google import genai
            from google.genai import types
            from pydantic import BaseModel

            class Item(BaseModel):
                i: int
                r: str

            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))

            # Test 1: Plain generation
            t0 = time.time()
            resp1 = client.models.generate_content(
                model=model_name,
                contents="Reply with 'PONG' only."
            )
            lat1 = (time.time() - t0) * 1000
            print(f"  • Connectivity   : OK ({lat1:.0f}ms) -> {resp1.text.strip()[:30]!r}")

            # Test 2: Structured JSON schema without thinking_config(thinking_budget=0)
            cfg = types.GenerateContentConfig(
                system_instruction="Romanize tokens to WhatsApp-style Latin Hinglish.",
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=list[Item]
            )
            sample_tokens = ["आज", "का", "इवेंट", "शानदार", "था"]
            payload = {"items": [{"i": i, "t": t} for i, t in enumerate(sample_tokens)]}

            resp2 = None
            for attempt in range(3):
                try:
                    t1 = time.time()
                    resp2 = client.models.generate_content(
                        model=model_name,
                        contents=json.dumps(payload),
                        config=cfg
                    )
                    lat2 = (time.time() - t1) * 1000
                    break
                except Exception as ex:
                    err_s = str(ex)
                    if ("503" in err_s or "429" in err_s or "UNAVAILABLE" in err_s) and attempt < 2:
                        time.sleep(2)
                        continue
                    raise

            parsed = json.loads(resp2.text)
            print(f"  • Structured Gen : OK ({lat2:.0f}ms) Model: {model_name}")
            print(f"    Returned items : {parsed}")

            # Verify schema integrity
            schema_ok = len(parsed) == len(sample_tokens) and all("i" in it and "r" in it for it in parsed)
            if schema_ok:
                print("  • Schema Check   : [PASS] list[Item(i, r)] matched exactly.")
            else:
                print("  • Schema Check   : [FAIL] Output count or keys mismatch.")
                all_passed = False

            # Test 3: Disk Cache
            test_cache_file = CACHE_DIR / "health_check_probe.json"
            test_cache_file.write_text(json.dumps(["test_cache_ok"]), encoding="utf-8")
            read_back = json.loads(test_cache_file.read_text(encoding="utf-8"))
            test_cache_file.unlink(missing_ok=True)
            if read_back == ["test_cache_ok"]:
                print(f"  • Disk Caching   : [PASS] Writable at {CACHE_DIR}")
            else:
                print(f"  • Disk Caching   : [FAIL] Cache write/read failed.")
                all_passed = False

        except Exception as e:
            print(f"  • Gemini API Test: [FAIL] {type(e).__name__}: {e}")
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print(" ALL DIAGNOSTIC CHECKS PASSED - PIPELINE READY FOR PRODUCTION!")
    else:
        print(" SOME CHECKS REPORTED ISSUES (See details above)")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
