# Physics Wallah (PW) AI Viral Reel Automation Pipeline - Final Benchmark & Performance Report

**Date**: November 8, 2026  
**Platform**: macOS Darwin (arm64, Apple Silicon M4, 16 GB Unified RAM)  
**Python Runtime**: Python 3.14.6  
**FFmpeg Binary**: FFmpeg 9.0.2 (Apple Silicon VideoToolbox, libx264, NEON)  
**AI Brain**: Google GenAI SDK (`gemini-3.5-flash-lite`)  
**Speech Engine**: MLX Whisper (`mlx-community/whisper-large-v3-mlx` on GPU & `base-mlx`)  

---

## 1. Executive Summary

Over the course of optimization iterations, the PW Reel Maker pipeline was overhauled from a multi-pass sequential script into a high-performance, single-pass fused rendering engine with centralized word caching, hardware-accelerated video encoding, and intelligent prosody-guided viral clip selection.

### Key Milestones Achieved:
1. **End-to-End Speedup**: **31.5x faster stage execution** (from **77.75s** down to **2.47s** on the 30s benchmark sample). Total CLI wall-clock time dropped from **89.50s** down to **4.0s** (**22.4x overall speedup**).
2. **Zero Redundant Whisper Transcription**: Word-level timestamps from the full-lecture scan are cached in persistent disk storage and sliced directly for boundary refinement and caption burning, eliminating up to two redundant ~43s model inference passes per clip.
3. **Unified Single-Pass Fused Render**: Replaced 3 lossy intermediate video encodes (cut $\to$ reframe $\to$ caption) with a single producer-consumer queue and a unified `cv2.warpAffine` + sprite ROI composite loop.
4. **Hardware VideoToolbox Encoding**: Final 1080x1920 9:16 export throughput increased from ~30 fps to **160–200 fps** via `h264_videotoolbox` (`--fast`).
5. **Sprite Cache & ROI Blending**: Replaced full-frame PIL conversions and alpha compositing with tight RGBA float32 premultiplied sprite blending (< 0.1 ms per frame).
6. **Strict Invariant Adherence**:
   - Audio-to-video drift: **0.0 ms** (threshold: $\le 50$ ms).
   - Latin-only ASCII guarantee: **100%** (zero tofu boxes $\square\square\square$).
   - Framing anti-corner safe zone: $[0.32, 0.68]$ strictly maintained.
   - ExtendScript JSX compatibility: Embedded native JavaScript data without `JSON.parse`.

---

## 2. Before vs. After Timing Profile Comparison

The following table compares the original sequential baseline profile recorded in [`docs/baseline_profile.md`](file:///Users/sandeep/Desktop/PW/docs/baseline_profile.md) against the optimized pipeline executing on the exact same benchmark sample (`samples/sample.mp4`, 25.0 fps, 1080x1920 target):

| Pipeline Stage | Baseline (s) | Optimized (s) | Speedup Factor | Mechanism & Optimization |
| :--- | :---: | :---: | :---: | :--- |
| **Snippet Cut** | 0.51s | **0.25s** | **2.0x** | High-speed copy / CRF 10 near-lossless cut with PTS zeroing |
| **Boundary Refine** | 2.77s | **0.04s** | **69.3x** | Slices cached word-level timestamps instead of launching Whisper |
| **Reframe (Pass 1)** | 10.04s | **0.67s** | **15.0x** | Haar on downscaled 320px gray frame every 3rd frame + template tracking |
| **Whisper Word Sync** | 43.30s | **0.00s** | **$\infty$ (Instant)** | Bypasses Whisper entirely via linear timestamp mapping from lecture cache |
| **Gemini Romanize** | 2.05s | **0.01s** | **205x (Cache)** | Persistent hash-keyed JSON cache + `gemini-3.5-flash-lite` |
| **Reframe Render** | 8.91s | — | — | *Eliminated* (Merged into unified render) |
| **Caption Burn** | 10.18s | — | — | *Eliminated* (Merged into unified render) |
| **Unified Single Render** | — | **1.51s** | **12.6x vs combined** | Fused `cv2.warpAffine` + sprite ROI blend + `h264_videotoolbox` @ 199.8 fps |
| **Total Stage Time** | **77.75s** | **2.47s** | **31.5x Speedup** | Sum of individual pipeline stages |
| **Total CLI Wall Time** | **89.50s** | **4.00s** | **22.4x Speedup** | Includes lecture scan, AI hook detection, and file I/O |

*Data recorded in `output/profile_1clip_opt/profile_reel_01_imandari_se_batana.json`.*

---

## 3. Multi-Clip End-to-End Performance (3 Clips on 117s Real Lecture)

To stress-test multi-clip throughput, boundary refinement, and AV synchronization on real-world classroom footage, the pipeline was executed end-to-end on `/Users/sandeep/Desktop/lecture_hook_25fps.mov` (116.7s duration, 854x480 source):

### Output Clips Summary:
1. **Reel #1: "Conditions Kabhi Nahi Sudrengi!"**
   - **File**: `output/final_3clips_opt/reel_01_conditions_kabhi_nahi_sudrengi.mp4`
   - **Duration**: 26.5s (662 frames @ 25 fps, 1080x1920)
   - **Render Rate**: 155.2 fps (4.27s total encode)
   - **AV Sync Diff**: **0.0 ms** (LOCKED)
   - **Captions**: 74 words, kinetic-pop with active spring bounce

2. **Reel #2: "Apna Future Imagine Karo!"**
   - **File**: `output/final_3clips_opt/reel_02_apna_future_imagine_karo.mp4`
   - **Duration**: 23.4s (584 frames @ 25 fps, 1080x1920)
   - **Render Rate**: 156.5 fps (3.73s total encode)
   - **AV Sync Diff**: **0.0 ms** (LOCKED)
   - **Captions**: 71 words, kinetic-pop with active spring bounce

3. **Reel #3: "Ab Baari Aapki Hai"**
   - **File**: `output/final_3clips_opt/reel_03_ab_baari_aapki_hai.mp4`
   - **Duration**: 16.0s (400 frames @ 25 fps, 1080x1920)
   - **Render Rate**: 156.9 fps (2.55s total encode)
   - **AV Sync Diff**: **0.0 ms** (LOCKED)
   - **Captions**: 5 words, kinetic-pop

### Multi-Clip Invariant Verification Table:

| Metric / Invariant | Target Threshold | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Audio-Video Lip Sync Drift** | $\le 50\text{ ms}$ | **0.0 ms** on all 3 reels | **PASS** |
| **Output Aspect Ratio & Resolution** | $1080 \times 1920$ (9:16) | $1080 \times 1920$ | **PASS** |
| **Framerate Mode** | Constant (CFR 25.0) | `cfr=True` | **PASS** |
| **Face Lock Ratio (talking_head)** | $\ge 85\%$ | **86.1% – 91.3%** | **PASS** |
| **Lossy Video Re-Encodes** | Exactly 1 lossy encode | **1** (`render_reel`) | **PASS** |
| **Latin-Only Characters** | 100% Latin ASCII | **100%** (0 tofu boxes) | **PASS** |

---

## 4. Subsystem Deep-Dive & Architecture Summary

### 4.1 AI Viral Hook Finder (`core/hook_finder.py`)
- Leverages **`gemini-3.5-flash-lite`** with native Pydantic structured schemas (`list[ViralClipCandidate]`).
- Segment IDs (`[S0001]`, `[S0002]`, etc.) are mapped to timestamps in Python to prevent LLM arithmetic hallucination.
- Incorporates acoustic features: 5s window RMS energy $z$-scores, words/sec speech rate, and pause density.
- Computes weighted multi-metric score:
  $$\text{Score} = 0.35 \cdot \text{hook} + 0.20 \cdot \text{standalone} + 0.15 \cdot \text{emotion} + 0.15 \cdot \text{payoff} + 0.15 \cdot \text{shareability}$$
- Evaluates teacher face visibility to drop blackboard-only segments.

### 4.2 Centralized Whisper Backend (`core/whisper_backend.py`)
- Detects hardware environment and automatically activates `mlx-whisper` on Apple Silicon GPU/Metal.
- Transcribes full lecture audio once with word-level timestamps and caches results on disk:
  $$\text{Key} = \text{SHA256}(\text{path} + \text{size} + \text{mtime} + \text{model})$$
- Linear time mapping for cut snippets:
  $$t_{\text{final}} = t_{\text{orig}} - t_{\text{clip\_start}} + \frac{\text{audio\_delay\_ms}}{1000}$$
- Falls back to a fresh local Whisper pass if word coverage in the cut window is $< 80\%$.

### 4.3 Pluggable Face Detector (`core/reframer.py`)
- Pluggable backends:
  - `haar`: OpenCV Haar cascade (`haarcascade_frontalface_default.xml`).
  - `yunet`: OpenCV Zoo ONNX model (`face_detection_yunet_2023mar.onnx`).
  - `vision`: Apple Vision framework via PyObjC.
- Detects every 3rd frame on a 320px downscaled image with intermediate optical template tracking.
- Static cluster rejection detects stationary slide/board elements via variance and motion thresholds.
- Back-turned motion fallback tracks motion centroids when the teacher turns toward the board.

### 4.4 Sprite Cache Captioner (`core/captioner.py`)
- Precomputes tight RGBA sprites for every `(chunk, active_word_idx, animation_step)` state once.
- Gaussian glow blur and strokes are computed inside the sprite bounding box rather than full canvas.
- Real-time frame compositing blends only the sprite ROI via premultiplied float32 BGR arrays in NumPy.
- 14 production styles supported, including `kinetic-pop` and `kinetic-roll`.

### 4.5 After Effects ExtendScript Integration (`core/export_ae.py`)
- Generates `.jsx` ExtendScript with data embedded as `var DATA = {...};` (no `JSON.parse`).
- Applies exact anchor-centered keyframe math:
  $$s(t) = \frac{\text{out\_h}}{\text{crop\_h}(t)}$$
  $$\text{Position} = \left(\frac{\text{out\_w}}{2} - \left(cx - \frac{W}{2}\right) \cdot s, \; \frac{\text{out\_h}}{2} - \left(cy - \frac{H}{2}\right) \cdot s\right)$$
  $$\text{Scale} = s \cdot 100$$
- Simplifies trajectories using RDP to 5–10 keyframes/sec while preserving scene cuts.
- Generates companion SubRip `.srt` and Premiere Pro markers `.csv`.

---

## 5. Security & Repository Hygiene
- **Zero Committed Secrets**: Comprehensive repo-wide search verified no hardcoded API keys exist.
- **Environment Driven**: Gemini API authentication strictly relies on `os.getenv("GEMINI_API_KEY")` or `--api-key`.
- **Dependency Pinning**: All dependencies are strictly pinned in `requirements.txt` and `pyproject.toml`.

---

## 6. Conclusion
The Physics Wallah Reel Maker pipeline is fully verified, thoroughly profiled, and production-ready for automated high-volume short-form content generation.

