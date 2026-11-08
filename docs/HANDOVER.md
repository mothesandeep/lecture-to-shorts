# Physics Wallah (PW) AI Viral Reel Automation Pipeline - Handover Document

**Version**: 2.0.0  
**Target Platform**: macOS Darwin (arm64, Apple Silicon M-Series / M4) & Linux/Windows (CUDA/CPU)  
**Primary AI Model**: `gemini-3.5-flash-lite` (Google GenAI SDK v3.0.0)  
**Speech Recognition Engine**: MLX Whisper (Apple Silicon GPU) & OpenAI Whisper (CPU fallback)

---

## 1. Executive Summary & Architecture Overview

The PW Reel Maker Pipeline transforms full-length classroom lectures into vertical 9:16 reels for social media platforms and generates After Effects ExtendScript JSX projects for professional editing workflows.

### High-Level Data Flow:
```
Lecture Video (.mp4 / .mov)
   │
   ├── [Stage 1: HookFinder & WhisperBackend]
   │     • MLX / OpenAI Whisper extracts word-level timestamps (stored in ~/.cache/hinglish_captions)
   │     • AudioFeatureExtractor computes energy z-scores, speech rate, pause density
   │     • gemini-3.5-flash-lite selects top viral hooks via structured JSON schemas
   │
   ├── [Stage 2: Boundary Refiner]
   │     • Snaps clip boundaries to natural speech pauses (<= 0.35s) using cached words
   │
   ├── [Stage 3: Snippet Cutter]
   │     • Cuts audio-synced snippet with 500 ms delay baked in and PTS zeroed (CRF 10)
   │
   ├── [Stage 4: FaceReframer.plan()]
   │     • Pluggable detector (Haar, YuNet, Vision) runs every Nth frame (default: 3)
   │     • Template tracker, static cluster rejection, back-turned motion fallback
   │     • Trajectory smoothing with safe-zone bounds [0.32, 0.68] & optional punch-in zoom
   │
   ├── [Stage 5: Transcriber & Romanizer]
   │     • Slices cached words (t_final = t_orig - t_start + audio_delay)
   │     • Romanizes Devanagari to Latin Hinglish via gemini-3.5-flash-lite / local rules
   │     • Enforces 100% Latin-only ASCII guarantee
   │
   └── [Stage 6: Unified Render OR AE Export]
         • Production: render_reel() fused single-pass warpAffine + sprite blend -> FFmpeg
         • Post-Production: export_ae_project() -> .jsx + .srt + .csv
```

---

## 2. Core Model Configurations & Contracts

### 2.1 Gemini LLM Model: `gemini-3.5-flash-lite`
> [!IMPORTANT]
> The primary multimodal and structured language model used throughout the entire pipeline is **`gemini-3.5-flash-lite`** (NOT Gemini 2.5 or earlier).
> - **API Client**: `google.genai.Client` with `google-genai==3.0.0`.
> - **Environment Variable**: `GEMINI_API_KEY` (override via CLI `--api-key`).
> - **Configuration**:
>   - `temperature`: `0.2` for hook detection, `0.0` for Hinglish transliteration.
>   - `response_mime_type`: `"application/json"`.
>   - `response_schema`: Strict Pydantic models (`list[ViralClipCandidate]`, `list[Item]`, `ViralClipMetadata`).
> - **Fallback**: If `GEMINI_API_KEY` is not present, `HookFinder` requires an explicit `--api-key` or user-supplied timestamps (`pipeline.py clip`), while `HinglishTranscriber` automatically falls back to offline rule-based phonetic transliteration.

### 2.2 Whisper Transcription Engine
- **Centralized Engine**: `core/whisper_backend.py`.
- **Backend Selection**:
  - `auto`: Uses `mlx` (`mlx-whisper`) if installed on Apple Silicon, otherwise `openai` (`openai-whisper`).
  - `mlx`: Hardware-accelerated on Apple Silicon GPU/Unified Memory (Metal).
  - `openai`: Standard PyTorch implementation on CPU.
- **Persistent Disk Cache**:
  - Cache key: `SHA256(filename + size + mtime + model)`.
  - Cache directory: `~/.cache/hinglish_captions/transcript_<hash>.json`.
  - Stores all words with float timestamps: `Word(start, end, text)`.
- **Decoding Invariants**:
  - `language`: `"hi"` (Hindi/Hinglish).
  - `no_speech_threshold`: `0.3`.
  - `condition_on_previous_text`: `False`.

---

## 3. Subsystem Invariants & Critical Guarantees

### 3.1 Audio-Video Synchronization & Delay Contract
1. **Baked-In Audio Delay**:
   - The instructor audio arrives ~500 ms early relative to video in raw classroom streams.
   - The delay is applied **strictly once** in `cut_snippet_without_reencode` / `cut_snippet_accurate` via FFmpeg audio filter `adelay=500|500`.
   - All subsequent downstream renderers (`render_reel`, `reframe_video`, `CaptionRenderer`) receive `audio_delay_ms = 0.0` so delay is never applied twice.
2. **PTS Zeroing & CFR**:
   - Every cut snippet resets timestamps: `setpts=PTS-STARTPTS` and `asetpts=PTS-STARTPTS`.
   - Video is normalized to Constant Frame Rate (CFR) via `-fps_mode cfr` and `aresample=async=1000`.
3. **Drift Threshold**:
   - `core/av_sync.py:verify_av_sync` ensures $|\Delta t| \le 50\text{ ms}$ between audio and video streams.

### 3.2 100% Latin-Only Caption Invariant
- Indian classroom typography renders on Western fonts (e.g. Arial, Impact, Montserrat). Any un-romanized Devanagari (`क`, `म`) or Perso-Arabic glyph results in missing glyph tofu boxes (`□□□`).
- `core/transcriber.py:sanitize_latin` strips any lingering non-ASCII unicode characters.
- Invariant: `assert DEV.search(sanitized) is None and sanitized.isascii()`.

### 3.3 Framing Anti-Corner Safe Zone Invariant
- Center of crop window $c_x(t) \in [0.32 \cdot W, 0.68 \cdot W]$ to guarantee the subject is never pushed into screen margins.
- Zoomed crops enforce $W_{\text{eff}} = W_{\text{base}} / z(t)$ and clamp the window entirely inside the source boundaries.

### 3.4 Single Lossy Encode (Fused Pipeline)
- Old architecture: snippet cut $\to$ reframe encode $\to$ caption encode (3 lossy re-encodes).
- Modern architecture: snippet cut (CRF 10 lossless/near-lossless) $\to$ **1 unified fused encode** via `core/render.py:render_reel`.

### 3.5 ExtendScript ES3 Compatibility
- ExtendScript engine in Adobe After Effects does NOT support standard `JSON.parse` or ES6 modules.
- `core/export_ae.py` writes data directly into the script as a native JavaScript literal:
  `var DATA = { ... };`
- Synced footage imported into AE is the 500 ms audio-delayed snippet, never the raw unsynced master.

---

## 4. Keyframe Math for After Effects

After Effects layers have their anchor point at layer center: $(W/2, H/2)$.
Given target canvas $1080 \times 1920$ and per-frame crop $(cx, cy, \text{crop\_w}, \text{crop\_h})$:
$$s(t) = \frac{\text{out\_h}}{\text{crop\_h}(t)}$$
$$\text{Position}_x(t) = \frac{\text{out\_w}}{2} - \left(cx(t) - \frac{W}{2}\right) \cdot s(t)$$
$$\text{Position}_y(t) = \frac{\text{out\_h}}{2} - \left(cy(t) - \frac{H}{2}\right) \cdot s(t)$$
$$\text{Scale}(t) = s(t) \cdot 100$$

Keyframes are simplified via **Ramer–Douglas–Peucker (RDP)** to 5–10 keyframes per second while strictly retaining scene cuts and sharp extrema.

---

## 5. Security & Maintenance

1. **API Keys**:
   - Strictly avoid hardcoding API keys in repository files or commit messages.
   - Use `export GEMINI_API_KEY="your-key"`.
2. **Rotating Keys**:
   - If rotating Gemini API keys in the future, set the new key in the environment or CI secret manager.
3. **Model Weights**:
   - OpenCV YuNet model is located at `models/face_detection_yunet_2023mar.onnx` (~232 KB).
   - If missing, re-fetch via `python3 scripts/download_models.py`.
4. **Verification**:
   - Run `python3 pipeline.py test` before pushing changes to production.
   - Run `python3 -m unittest discover tests` to ensure all 65+ unit tests pass.

