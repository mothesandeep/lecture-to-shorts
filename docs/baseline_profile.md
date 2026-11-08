# Pipeline Baseline Timing Profile

## 1. Environment & Hardware
- **Platform**: macOS Darwin (arm64, Apple Silicon M4, 16 GB Unified RAM)
- **Python Runtime**: Python 3.14.6
- **FFmpeg Binary**: FFmpeg 9.0.2 with libx264, Videotoolbox, NEON optimizations
- **Input Sample**: `samples/sample.mp4` (30.0s cut via `scripts/make_sample.sh` from lecture footage)
- **Processed Reel**: `output/reel_01_selection_wali_feeling.mp4` (29.4s after boundary refinement)

## 2. Command Executed
```bash
python3 pipeline.py run \
  --input samples/sample.mp4 \
  --clips 1 \
  --min-duration 15.0 \
  --max-duration 30.0 \
  --refine-boundaries \
  --profile
```

## 3. Per-Stage Wall-Clock Timing Breakdown

| Stage | Wall-Clock Time (s) | Share of Stage Time (%) | Description |
| :--- | :---: | :---: | :--- |
| **cut** | 0.51s | 0.7% | Audio delay (500 ms) lip-sync offset + PTS zeroing snippet cut |
| **boundary refine** | 2.77s | 3.6% | +/-60s speech pause scanning for natural sentence boundary |
| **reframe pass 1** | 10.04s | 12.9% | Haar cascade face detection, static slide rejection, and trajectory smoothing |
| **reframe render** | 8.91s | 11.5% | FFmpeg rawvideo pipe rendering & encoding to 1080x1920 9:16 vertical video |
| **whisper** | 43.30s | 55.7% | Loading `large-v3` + word-level timestamps extraction |
| **gemini romanize** | 2.05s | 2.6% | Structured Pydantic schema Hinglish transliteration via `gemini-3.5-flash-lite` |
| **caption burn** | 10.18s | 13.1% | Pillow rendering with active word animation & FFmpeg pipe encoding |
| **Total Stage Time** | **77.75s** | **100.0%** | Sum of individual clip pipeline stages |
| **Total Wall Time** | **89.50s** | — | Total CLI execution including full-lecture scan & hook detection |

*Raw data serialized to: `output/profile_reel_01_selection_wali_feeling.json` and `output/profile_01_selection_wali_feeling.json`.*

## 4. Hardware Acceleration & Whisper Analysis

### Device Execution
- `torch.backends.mps.is_available()`: `True`
- `model.device`: **`cpu`**
- **Root Cause**: OpenAI Whisper's `load_model(name, device=None)` checks only `torch.cuda.is_available()`. On Apple Silicon macOS, MPS is available in PyTorch, but Whisper falls back to `cpu` unless explicitly passed `device="mps"`. Furthermore, Whisper issues `UserWarning: FP16 is not supported on CPU; using FP32 instead`, running full FP32 math on the CPU cores.

### Model Loading Frequency Per Clip
- **Full lecture scan**: 1 load (`whisper_full_model`, default: `base`) in `HookFinder.transcribe_full_audio`.
- **Boundary refinement**: 1 load (`whisper_model`, default: `base`) in `refine_clip_boundaries`.
- **Word-level sync**: 1 load (`whisper_clip_model`, default: `large-v3`, ~3 GB weights) in `HinglishTranscriber.transcribe_and_romanize`.
- **Total per clip**: **2 Whisper model loads per clip** (or **3 total loads** for a 1-clip pipeline run including lecture hook detection).
- Neither model instance is cached in memory across stages or across multiple clips, meaning every clip reloads model weights from disk into RAM.

## 5. Invariant Verifications
1. **Latin-Only Captions**: Strictly maintained. Zero Devanagari or Perso-Arabic characters reached Pillow rendering; all 88 tokens transliterated to clean Latin ASCII.
2. **Audio Delay**: Applied exactly once in `av_sync.cut_snippet_accurate` (500 ms). `reframe_video` received `audio_delay_ms=0.0`.
3. **AV Sync**: `verify_av_sync` measured `diff=0.0ms` (threshold $\le 50\,\text{ms}$) on both reframed and final captioned reels.
4. **Framing Invariant**: Anti-corner safe zone $[0.32, 0.68]$ remained enforced throughout all 736 frames.
