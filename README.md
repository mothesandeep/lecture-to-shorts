# Physics Wallah (PW) AI Viral Reel Automation Pipeline

An enterprise-grade, GPU-accelerated video automation pipeline designed to convert full-length educational lectures into viral, high-retention 9:16 vertical short-form reels (Instagram Reels, YouTube Shorts, TikTok) and professional After Effects / Premiere Pro project files.

Built and optimized for **Apple Silicon (M-Series, M4)** and cross-platform production.

---

## ⚡ Key Highlights & Architecture

1. **AI Viral Hook Detection (`gemini-3.5-flash-lite`)**
   - Transcribes lectures once and enriches transcripts with multi-modal prosody features (RMS energy $z$-scores, words/sec speech rate, pause density).
   - Prompts Google's **`gemini-3.5-flash-lite`** using strict Pydantic structured JSON schemas.
   - Computes weighted sub-scores (hook strength, standalone context, emotional valence, payoff, shareability) to rank clips.
   - Automatic Map-Reduce sliding window with overlap for long (60+ min) lectures.
   - Teacher face presence filter ($\ge 70\%$ threshold) drops un-reframeable blackboard/slide-only clips.

2. **Smart Boundary Refinement**
   - Snaps clip start/end boundaries to natural sentence pauses ($\le 0.35$s threshold) using cached word timestamps.
   - Eliminates mid-sentence cuts and awkward syllable cutoffs.

3. **Centralized Word-Level Whisper Cache**
   - Full-lecture Whisper pass stores word-level timestamps in persistent disk cache keyed by `(filename + size + mtime + model)`.
   - Clip boundary snapping and caption generation reuse sliced word timestamps with linear time mapping ($t_{\text{final}} = t_{\text{orig}} - t_{\text{clip\_start}} + \text{audio\_delay}$).
   - Zero redundant re-transcription. Gracefully falls back to a fresh pass if speech coverage in slice is $< 80\%$.

4. **Pluggable Vertical Reframer**
   - Pluggable face detectors: **Haar Cascade** (default, CPU), **YuNet ONNX** (OpenCV Zoo neural detector), or **Apple Vision** (PyObjC hardware-accelerated).
   - Fast 3-frame downscaled detection with intermediate template tracking for high throughput.
   - Static cluster rejection rejects static whiteboard diagrams, logos, and slides.
   - Back-turned fallback leverages motion mask centroids and temporal hold when the instructor faces the board.
   - Optional punch-in zoom triggered by audio emphasis peaks with smoothstep easing.

5. **Unified Single-Pass Fused Renderer**
   - Replaces traditional multi-step intermediate encodes (cut $\to$ reframe $\to$ caption) with **one single-pass render**.
   - Single affine warp (`cv2.warpAffine`) applies crop, scale, and zoom in a single interpolation step.
   - Multi-threaded producer-consumer queue (`queue.Queue(maxsize=8)`) feeds uncompressed rawvideo frames directly to FFmpeg `stdin`.
   - Hardware-accelerated encoding via Apple Silicon VideoToolbox (`--fast`, `h264_videotoolbox -b:v 12M`) or CPU `libx264`.
   - Guaranteed exact duration, CFR (-fps_mode cfr), and strict AV lip-sync ($\le 50$ms drift).

6. **Punchy Kinetic Captions & Fast Sprite Blending**
   - Pre-renders each caption word state **once** into a tight RGBA sprite ROI with float32 premultiplied alpha.
   - Blends sprite ROIs directly into video frames via vectorized NumPy operations ($< 0.1$ ms per frame, zero full-frame PIL overhead).
   - 14 distinct visual styles, including `kinetic-pop` (entrance pop + active spring bounce) and `kinetic-roll` (2-line rolling layout).
   - **100% Latin-Only Guarantee**: Sanitizes Hindi/Hinglish tokens into pure Latin ASCII, completely eliminating tofu box (`□□□`) font errors.

7. **After Effects & Premiere Pro Export (`pipeline.py export`)**
   - Automatically generates ExtendScript `.jsx` files with embedded data (`var DATA = {...};`, strictly avoiding `JSON.parse` for ES3 compatibility).
   - Imports pre-cut audio-synced footage and applies layer-center Position/Scale keyframes.
   - Simplifies motion keyframes using Ramer–Douglas–Peucker (RDP) to 5–10 keyframes/sec so human video editors can adjust trajectories.
   - Exports SubRip `.srt` subtitles and SMPTE timecode markers CSV.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.10 – 3.14 (Python 3.14 on macOS arm64 recommended).
- **FFmpeg & FFprobe**: Must be available on system PATH (`brew install ffmpeg` on macOS).
- **macOS Apple Silicon**: M1/M2/M3/M4 for MLX Whisper and VideoToolbox hardware acceleration.

### 2. Install Python Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd PW

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials
The pipeline uses **`gemini-3.5-flash-lite`** for high-speed AI hook detection and Hinglish transliteration:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
*(Note: If no API key is provided, the pipeline gracefully falls back to local rule-based phonetic transliteration).*

### 4. Optional Pre-trained Vision Models
To enable OpenCV Zoo YuNet neural face detection:
```bash
python3 scripts/download_models.py
```

### 5. Run Health Check
Verify all system drivers, hardware encoders, Whisper backends, detectors, and Gemini structured output:
```bash
python3 pipeline.py test
```

---

## 🚀 CLI Command Reference

### 1. Automated Lecture-to-Reels (`pipeline.py run`)
Processes an entire lecture, finds top viral hooks, reframes to 9:16 vertical, generates synchronized kinetic captions, and exports finalized MP4 reels:
```bash
python3 pipeline.py run \
  --input samples/sample.mp4 \
  --clips 3 \
  --style kinetic-pop \
  --detector haar \
  --fast \
  --output-dir output/
```

Key Options:
- `--fast`: Use Apple Silicon VideoToolbox (`h264_videotoolbox -b:v 12M`) for 10–15x real-time rendering.
- `--style`: Caption style (`kinetic-pop`, `kinetic-roll`, `classic-yellow`, `ae-glow`, `box-highlight`, etc.).
- `--detector`: Face detector backend (`haar`, `yunet`, `vision`).
- `--punch-in`: Dynamic punch-in zoom (`off`, `auto`).
- `--whisper-backend`: `auto` (uses `mlx` on Apple Silicon GPU, `openai` on CPU).
- `--profile`: Prints detailed stage-by-stage wall-clock performance metrics.

### 2. Manual Timestamp Cut (`pipeline.py clip`)
Renders a specific time window:
```bash
python3 pipeline.py clip \
  --input samples/sample.mp4 \
  --start 10.0 \
  --end 38.5 \
  --aspect 9:16 \
  --style kinetic-pop \
  --fast
```

### 3. After Effects ExtendScript Export (`pipeline.py export`)
Generates an After Effects project bundle for human editors:
```bash
python3 pipeline.py export \
  --input samples/sample.mp4 \
  --clip 1 \
  --output-dir output/ae_export/
```
To import into After Effects:
1. Open Adobe After Effects.
2. Select **File → Scripts → Run Script File...**
3. Choose the generated `.jsx` file.

### 4. Hook Detection Only (`pipeline.py detect`)
Scans lecture and outputs ranked hook proposals without rendering:
```bash
python3 pipeline.py detect \
  --input samples/sample.mp4 \
  --clips 5 \
  --json-out output/hooks.json
```

### 5. Standalone Reframing (`pipeline.py reframe`)
Executes camera tracking without captions:
```bash
python3 pipeline.py reframe \
  --input samples/sample.mp4 \
  --preset talking_head \
  --detector haar
```

### 6. Standalone Captions (`pipeline.py caption`)
Burns animated captions onto an existing video:
```bash
python3 pipeline.py caption \
  --input output/reframed.mp4 \
  --style kinetic-roll
```

---

## 🧪 Testing & Verification

Run the comprehensive unit test suite:
```bash
python3 -m unittest discover tests
```
The test suite covers:
- Centralized Whisper caching, hash keys, and word timestamp slicing.
- Single-pass fused renderer, CFR enforcement, and AV sync verification.
- Fast sprite ROI blending and pixel equivalence tests.
- Pluggable detectors (Haar, YuNet, Vision) and motion fallbacks.
- Punch-in zoom smoothstep ease curves and safe-zone invariants.
- Kinetic animation springs and greedy chunkers.
- After Effects ExtendScript JSX generation, math, and `node --check` syntax validation.
- End-to-end health check diagnostics with mocked API clients.

---

## 🔒 Security & Invariants
- **API Keys**: No hardcoded API keys exist in the repository. All secrets are loaded strictly from the `GEMINI_API_KEY` environment variable or `--api-key` CLI flag.
- **Audio Lip-Sync**: 500 ms audio delay is baked in once during snippet cutting with PTS zeroing; `verify_av_sync` verifies AV drift is $\le 50$ms.
- **Framing Invariant**: Anti-corner safe zone $[0.32, 0.68]$ is strictly maintained across all frames.
- **Tofu-Box Protection**: Latin sanitizer strips non-ASCII glyphs before Pillow text composition.

