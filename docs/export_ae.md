# Adobe After Effects Export & ExtendScript Integration Guide

## Overview

The `core/export_ae.py` module and `python3 pipeline.py export --clip N` CLI provide automated export of reframed lecture clips directly into Adobe After Effects ExtendScript (`.jsx`), SubRip subtitles (`.srt`), and timeline markers (`.csv`).

This workflow bridges autonomous AI viral clipping with high-end human post-production in After Effects, letting editors tweak camera moves, adjust keyframe easing, modify typography, and add bespoke motion graphics without having to recreate tracking or transcription from scratch.

---

## Export Bundle Contents

Each export generates a self-contained directory containing four synchronized assets:

| File | Purpose | Details |
|---|---|---|
| `<slug>_ae_setup.jsx` | ExtendScript automation script | Embeds native `var DATA = {...};` (no `JSON.parse`), creates 1080x1920 9:16 comp, applies Position & Scale keyframes, adds text layers and markers. |
| `<slug>.srt` | SubRip subtitle track | Industry-standard SRT subtitle file with millisecond-accurate word/chunk timestamps. |
| `<slug>_markers.csv` | Timeline markers CSV | Importable into Premiere Pro and After Effects marker panels (SMPTE timecodes + seconds). |
| `snippet_<slug>.mp4` | Audio-synced source footage | High-quality (CRF 10) trimmed snippet with the 500 ms audio delay baked in. |

> [!IMPORTANT]
> **Audio Lip-Sync Guarantee**: The export script always imports the cut snippet (`snippet_<slug>.mp4`) with the 500 ms audio lip-sync delay pre-applied. It never imports the raw unsynced lecture source, preventing audio drift across NLE timelines.

---

## Camera Geometry & Transform Math

After Effects assigns the layer center `(W / 2, H / 2)` as the default layer anchor point. To map the face-centered crop box `(cx, cy, crop_w, crop_h)` from source footage of size `W x H` onto an output comp of size `out_w x out_h` (1080x1920):

$$s(t) = \frac{\text{out\_h}}{\text{crop\_h}(t)}$$

$$\text{Position}_X(t) = \frac{\text{out\_w}}{2} - \left( cx(t) - \frac{W}{2} \right) \cdot s(t)$$

$$\text{Position}_Y(t) = \frac{\text{out\_h}}{2} - \left( cy(t) - \frac{H}{2} \right) \cdot s(t)$$

$$\text{Scale}(t) = s(t) \times 100\%$$

When the face tracking is centered $(cx = W/2, cy = H/2)$, Position evaluates exactly to $(540, 960)$—the center of the 1080x1920 comp. When the teacher moves to the right in the source ($cx > W/2$), the layer shifts left, placing the teacher in the center of the vertical frame.

---

## Keyframe Simplification (Ramer-Douglas-Peucker)

Raw face tracking produces keyframes on every frame (25–30 keyframes per second). Having keyframe dots on every single frame makes manual adjustment in the After Effects Graph Editor difficult.

`export_ae.py` simplifies the trajectory using an adaptive **Ramer-Douglas-Peucker (RDP)** algorithm:
- Simplifies position and scale down to **roughly 5–10 keyframes per second**.
- Preserves all significant inflection points, trajectory bends, and camera pans.
- **Shot cut preservation**: Camera cuts (scene cuts) are partitioned; keyframes at cut boundaries are strictly retained with no cross-shot interpolation.
- Editors can select and nudge individual keyframes without fighting an overwhelming dense wall of points.

---

## ExtendScript ES3 Compatibility Invariant

Adobe After Effects uses an older ECMAScript 3 (ES3) ExtendScript engine that lacks native `JSON.parse`.

- The export script writes the tracking and caption data as a **native JavaScript object literal**:
  ```javascript
  var DATA = {
    "compName": "Reel_9x16_01_imandari_se_batana",
    "compWidth": 1080,
    "compHeight": 1920,
    ...
  };
  ```
- **Strictly zero `JSON.parse`**: Evaluates directly without third-party JSON polyfills or `json2.js`.
- **Node.js Validation**: Automatically checked before saving using `node --check --input-type=commonjs` to guarantee valid syntax.

---

## Step-by-Step Manual Testing in Adobe After Effects

Follow these steps to test and verify the exported JSX script inside Adobe After Effects:

### 1. Generate the Export Bundle
Run the pipeline export command from your terminal:
```bash
python3 pipeline.py export --input samples/sample.mp4 --clip 1 --output-dir output/ae_export
```
This produces:
- `output/ae_export/9x16_01_imandari_se_batana_ae_setup.jsx`
- `output/ae_export/9x16_01_imandari_se_batana.srt`
- `output/ae_export/9x16_01_imandari_se_batana_markers.csv`
- `output/ae_export/snippet_9x16_01_imandari_se_batana.mp4`

### 2. Launch Adobe After Effects
- Open Adobe After Effects (CC 2020 through 2026+).
- Create a new project (`Cmd+N` / `Ctrl+N`) or open an existing project.

### 3. Run the ExtendScript
- In the top menu bar, select:
  **`File`** > **`Scripts`** > **`Run Script File...`**
- In the file dialog, navigate to your export directory (`output/ae_export/`) and select:
  **`9x16_01_imandari_se_batana_ae_setup.jsx`**
- Click **Open**.

### 4. Verify in the After Effects Timeline
Once the script executes, verify the following elements:

1. **Composition Created**:
   - Check the **Project Panel**: a new composition named `Reel_9x16_...` appears.
   - Comp settings: **1080 x 1920** (9:16 vertical), **25.0 fps**, square pixels (`1.0`).
2. **Footage Layer & Animation**:
   - The footage layer (`snippet_...mp4`) is automatically placed at the bottom of the layer stack.
   - Select the footage layer and press **`U`** to reveal all animated keyframes.
   - Inspect the **Position** and **Scale** tracks: keyframes are cleanly spaced roughly 5–10 per second.
   - Scrub through the timeline: the camera smoothly tracks the teacher with no abrupt jumps.
3. **Caption Word Layers**:
   - Individual text layers are stacked above the footage.
   - Each layer is named `[index]_[word]` (e.g. `1_IMANDARI`, `2_SE`, `3_BATANA`).
   - Check layer duration: each layer's `inPoint` and `outPoint` correspond to speech timing.
   - Text styling: Yellow fill with black stroke outline centered at `Y = 1540` (`1920 - 380px` bottom offset). Emphasis words are highlighted in coral red.
4. **Timeline Markers**:
   - Look at the comp ruler at the top of the timeline: word sync markers appear at every spoken word timestamp for rapid navigation.
5. **Single-Step Undo**:
   - Press **`Cmd+Z`** (`Ctrl+Z` on Windows). Notice that the entire composition, import, keyframes, and layers are undone in one single step (powered by `app.beginUndoGroup`).

---

## Working with Premiere Pro

If editing in Adobe Premiere Pro:

1. **SRT Subtitle Track**:
   - In Premiere Pro, choose **`File`** > **`Import...`** and select the `.srt` file.
   - Drag the subtitle track directly onto your sequence timeline.
   - Style fonts and position in the **Essential Graphics / Text** panel.
2. **Markers CSV**:
   - In Premiere Pro, open the **Markers** panel (`Window` > `Markers`).
   - Right-click in the Markers panel and choose **`Import Markers...`** to import `_markers.csv`.

---

## CLI Command Reference

### Export Clip #1 from Automatic Hook Detection:
```bash
python3 pipeline.py export --input samples/sample.mp4 --clip 1
```

### Export Clip #2 with Custom Preset & Target Keyframe Rate:
```bash
python3 pipeline.py export \
  --input samples/sample.mp4 \
  --clip 2 \
  --reframe-preset talking_head \
  --detector haar \
  --target-rate 6.0 \
  --output-dir output/ae_export_clip2
```

### Export a Manual Timestamp Window:
```bash
python3 pipeline.py export \
  --input samples/sample.mp4 \
  --start 10.0 \
  --end 40.0 \
  --title "Organic Chemistry Mechanism" \
  --output-dir output/ae_export_custom
```

### Export Clean Video Trajectory (No Captions):
```bash
python3 pipeline.py export \
  --input samples/sample.mp4 \
  --clip 1 \
  --no-caption
```
