#!/usr/bin/env python3
"""
export_ae.py - After Effects ExtendScript JSX project and asset exporter.

Generates:
1. ExtendScript (.jsx) with precomputed camera trajectory and word-level text layers.
   - Embedded data as literal JavaScript: `var DATA = {...};` (strictly no JSON.parse).
   - Imports the already audio-synced cut snippet (500ms delay baked in), never the original source.
   - Creates a 1080x1920 9:16 vertical composition at clip fps and duration.
   - Sets Position and Scale keyframes on the footage layer using setValuesAtTimes.
     Layer center anchor math:
       s = out_h / crop_h(t)
       Position = (out_w/2 - (cx - W/2)*s, out_h/2 - (cy - H/2)*s)
       Scale = s * 100
   - Keyframes simplified via Ramer-Douglas-Peucker (RDP) down to roughly 5-10 per second
     so human editors can easily inspect and nudge keyframes.
   - Creates one text layer per caption word with exact inPoint and outPoint.
   - Creates composition timeline markers for every spoken word.
2. SubRip subtitle file (.srt).
3. Markers CSV file for Premiere Pro / After Effects timeline import.
4. JS syntax validation via `node --check` if Node.js is installed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from core.reframer import ReframePlan
    from core.transcriber import WordCaption
except ImportError:
    from reframer import ReframePlan
    from transcriber import WordCaption


# ── Ramer-Douglas-Peucker (RDP) Keyframe Simplification ─────────────────────────────

def rdp_indices(points: np.ndarray, epsilon: float) -> List[int]:
    """
    Ramer-Douglas-Peucker curve simplification algorithm.
    Recursively prunes points that deviate less than epsilon from the line segment.
    Returns list of preserved point indices.
    """
    if len(points) <= 2:
        return list(range(len(points)))

    p1 = points[0]
    p2 = points[-1]
    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq == 0:
        distances = np.linalg.norm(points - p1, axis=1)
    else:
        # Perpendicular distance in N-dimensions
        t = np.dot(points - p1, line_vec) / line_len_sq
        t = np.clip(t, 0.0, 1.0)[:, np.newaxis]
        proj = p1 + t * line_vec
        distances = np.linalg.norm(points - proj, axis=1)

    max_idx = int(np.argmax(distances))
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        left = rdp_indices(points[:max_idx + 1], epsilon)
        right = rdp_indices(points[max_idx:], epsilon)
        return left[:-1] + [i + max_idx for i in right]
    else:
        return [0, len(points) - 1]


def simplify_keyframes(
    positions: np.ndarray,
    scales: np.ndarray,
    fps: float,
    min_rate: float = 5.0,
    max_rate: float = 10.0,
    cuts: Optional[List[int]] = None,
) -> List[int]:
    """
    Simplify dense per-frame trajectory keyframes to roughly 5-10 keyframes per second.
    Preserves exact shot cuts and inflection points while keeping keyframes editor-tweakable.
    """
    num_frames = len(positions)
    if num_frames <= 2:
        return list(range(num_frames))

    # Weight position (pixels) and scale (normalized so 10% scale change ~ 5px)
    norm_scales = scales[:, np.newaxis] * 0.5
    feature_points = np.hstack([positions, norm_scales])

    # Partition by scene cut boundaries to prevent smoothing across cuts
    if cuts is None:
        seg_cuts = [0, num_frames]
    else:
        valid_cuts = sorted(list(set([c for c in cuts if 0 < c < num_frames])))
        seg_cuts = [0] + valid_cuts + [num_frames]

    target_mid_rate = (min_rate + max_rate) / 2.0
    all_indices: set[int] = set()

    for s_start, s_end in zip(seg_cuts[:-1], seg_cuts[1:]):
        seg_pts = feature_points[s_start:s_end]
        seg_n = len(seg_pts)
        if seg_n <= 2:
            all_indices.update([s_start, max(s_start, s_end - 1)])
            continue

        seg_dur = seg_n / fps
        target_kf = max(2, int(round(target_mid_rate * seg_dur)))

        # Binary search for optimal epsilon that gives roughly target_kf count
        low_eps, high_eps = 0.005, 50.0
        best_idx = list(range(seg_n))
        best_diff = float("inf")

        for _ in range(16):
            mid_eps = (low_eps + high_eps) / 2.0
            idx = rdp_indices(seg_pts, mid_eps)
            diff = abs(len(idx) - target_kf)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

            if len(idx) > target_kf:
                low_eps = mid_eps
            else:
                high_eps = mid_eps

        # Prevent excessively large time gaps in steady movement
        max_frame_gap = max(2, int(round(fps / min_rate)))
        sorted_sub = sorted(best_idx)
        dense_sub: List[int] = []
        for i in range(len(sorted_sub) - 1):
            dense_sub.append(sorted_sub[i])
            gap = sorted_sub[i + 1] - sorted_sub[i]
            if gap > max_frame_gap:
                num_inter = gap // max_frame_gap
                step = gap / (num_inter + 1)
                for k in range(1, num_inter + 1):
                    dense_sub.append(int(round(sorted_sub[i] + k * step)))
        dense_sub.append(sorted_sub[-1])

        # If too dense, prune with small step filter
        min_frame_gap = max(1, int(round(fps / (max_rate * 1.5))))
        pruned_sub: List[int] = []
        for k in dense_sub:
            if not pruned_sub or (k - pruned_sub[-1]) >= min_frame_gap:
                pruned_sub.append(k)
        if pruned_sub[-1] != (seg_n - 1):
            pruned_sub.append(seg_n - 1)

        for k in pruned_sub:
            all_indices.add(s_start + k)

    return sorted(list(all_indices))


# ── Time and Subtitle Formatters ───────────────────────────────────────────────────

def format_srt_time(seconds: float) -> str:
    """Format seconds into SubRip timestamp: HH:MM:SS,mmm."""
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hour = total_min // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def format_timecode(seconds: float, fps: float = 25.0) -> str:
    """Format seconds into SMPTE timecode: HH:MM:SS:FF."""
    fps_val = max(1.0, fps)
    total_frames = int(round(max(0.0, seconds) * fps_val))
    frames = int(total_frames % round(fps_val))
    total_sec = int(total_frames // round(fps_val))
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hour = total_min // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d}:{frames:02d}"


def generate_srt(words: List[WordCaption], out_path: Path) -> Path:
    """Export clean SubRip .srt subtitle file from word captions."""
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for idx, w in enumerate(words, start=1):
        s_time = format_srt_time(w.start)
        e_time = format_srt_time(w.end)
        lines.append(f"{idx}\n{s_time} --> {e_time}\n{w.word}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_markers_csv(words: List[WordCaption], fps: float, out_path: Path) -> Path:
    """Export timeline markers CSV for Premiere Pro / After Effects."""
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Marker Name",
            "In",
            "Out",
            "Duration",
            "Marker Type",
            "In Seconds",
            "Out Seconds",
            "Duration Seconds",
            "Description"
        ])
        for w in words:
            dur = max(0.04, w.end - w.start)
            writer.writerow([
                w.word,
                format_timecode(w.start, fps),
                format_timecode(w.end, fps),
                format_timecode(dur, fps),
                "Comment",
                f"{w.start:.4f}",
                f"{w.end:.4f}",
                f"{dur:.4f}",
                "Emphasis Word" if getattr(w, "emphasis", False) else "Transcript Word"
            ])

    return out_path


# ── JS Syntax Validation ───────────────────────────────────────────────────────────

def validate_jsx_syntax(jsx_path: Path) -> Tuple[bool, str]:
    """Validate ExtendScript / JS syntax using node --check if node is available."""
    node_bin = shutil.which("node")
    if not node_bin:
        return True, "Node.js not installed: skipped syntax check."

    code = jsx_path.read_text(encoding="utf-8")
    proc = subprocess.run(
        [node_bin, "--check", "--input-type=commonjs"],
        input=code,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip()
    return True, "Valid JavaScript syntax verified via node --check."


# ── ExtendScript JSX Generator ─────────────────────────────────────────────────────

def generate_extendscript_jsx(data: Dict[str, Any], out_path: Path) -> Path:
    """
    Generate an ExtendScript .jsx file with embedded DATA object literal.
    Strictly avoids JSON.parse for full compatibility with Adobe ExtendScript ES3 engine.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_json_literal = json.dumps(data, indent=2)

    script_content = f"""// After Effects ExtendScript Automation
// Generated by PW Reel Maker (export_ae.py)
// DATA is embedded directly as a native JavaScript object literal for ExtendScript compatibility.

var DATA = {data_json_literal};

(function() {{
    if (typeof app === "undefined" || !app.project) {{
        alert("Please open Adobe After Effects before running this script.");
        return;
    }}

    app.beginUndoGroup("Import PW Reel to Comp");
    try {{
        var project = app.project;

        // 1. Create Composition
        var comp = project.items.addComp(
            DATA.compName,
            DATA.compWidth,
            DATA.compHeight,
            1.0,
            DATA.duration,
            DATA.fps
        );
        comp.openInViewer();

        // 2. Locate and Import Source Footage
        var footageFile = new File(DATA.footagePath);
        if (!footageFile.exists && typeof $.fileName !== "undefined") {{
            var scriptFolder = (new File($.fileName)).parent;
            var relFile = new File(scriptFolder.fsName + "/" + DATA.footageFilename);
            if (relFile.exists) {{
                footageFile = relFile;
            }}
        }}

        if (!footageFile.exists) {{
            alert("Warning: Footage file not found at:\\n" + DATA.footagePath + "\\n\\nPlease link footage manually.");
        }} else {{
            var importOptions = new ImportOptions(footageFile);
            var footageItem = project.importFile(importOptions);
            var footageLayer = comp.layers.add(footageItem);
            footageLayer.name = DATA.footageFilename;
            footageLayer.startTime = 0;

            // 3. Set Position and Scale Keyframes
            var posProp = footageLayer.property("ADBE Transform Group").property("ADBE Position") || footageLayer.property("Position");
            var scaleProp = footageLayer.property("ADBE Transform Group").property("ADBE Scale") || footageLayer.property("Scale");

            if (posProp && scaleProp && DATA.keyframes.times.length > 0) {{
                posProp.setValuesAtTimes(DATA.keyframes.times, DATA.keyframes.positions);
                scaleProp.setValuesAtTimes(DATA.keyframes.times, DATA.keyframes.scales);

                // Set spatial interpolation to linear for predictable tracking
                for (var k = 1; k <= posProp.numKeys; k++) {{
                    try {{
                        posProp.setSpatialContinuousAtKey(k, true);
                        posProp.setTemporalContinuousAtKey(k, true);
                    }} catch (e) {{}}
                }}
            }}
        }}

        // 4. Create Caption Text Layers
        for (var i = 0; i < DATA.words.length; i++) {{
            var w = DATA.words[i];
            var textLayer = comp.layers.addText(w.text);
            textLayer.name = (i + 1) + "_" + w.text;
            textLayer.inPoint = w.start;
            textLayer.outPoint = w.end;

            // Position at horizontal center and bottom offset
            var tPos = textLayer.property("ADBE Transform Group").property("ADBE Position") || textLayer.property("Position");
            if (tPos) {{
                tPos.setValue([DATA.compWidth / 2, DATA.compHeight - DATA.captionConfig.bottomOffset]);
            }}

            // Style character and paragraph properties
            var tProp = textLayer.property("ADBE Text Properties").property("ADBE Text Document") || textLayer.property("Source Text");
            if (tProp) {{
                var td = tProp.value;
                td.fontSize = DATA.captionConfig.fontSize;
                td.font = DATA.captionConfig.fontFamily;
                td.applyFill = true;
                td.fillColor = w.isEmphasis ? [1.0, 0.32, 0.32] : DATA.captionConfig.fillColor;
                td.applyStroke = true;
                td.strokeColor = DATA.captionConfig.strokeColor;
                td.strokeWidth = DATA.captionConfig.strokeWidth;
                td.strokeOverFill = false;

                try {{
                    if (typeof ParagraphJustification !== "undefined") {{
                        td.justification = ParagraphJustification.CENTER_JUSTIFY;
                    }}
                }} catch (e) {{}}

                tProp.setValue(td);
            }}
        }}

        // 5. Add Timeline Markers for Word Sync Navigation
        if (comp.markerProperty) {{
            for (var m = 0; m < DATA.words.length; m++) {{
                var wm = DATA.words[m];
                var markerVal = new MarkerValue(wm.text);
                markerVal.duration = Math.max(0.04, wm.end - wm.start);
                comp.markerProperty.setValueAtTime(wm.start, markerVal);
            }}
        }}

    }} catch (err) {{
        alert("Error executing ExtendScript:\\n" + err.toString());
    }} finally {{
        app.endUndoGroup();
    }}
}})();
"""
    out_path.write_text(script_content, encoding="utf-8")
    return out_path


# ── Main Export Orchestrator ───────────────────────────────────────────────────────

def export_ae_project(
    snippet_path: Union[str, Path],
    plan: ReframePlan,
    words: List[WordCaption],
    out_dir: Union[str, Path] = "output/ae_export",
    comp_name: Optional[str] = None,
    caption_bottom_offset: int = 380,
    caption_font_size: int = 56,
    target_rate: float = 7.0,
    copy_snippet: bool = True,
) -> Dict[str, Path]:
    """
    Export complete After Effects ExtendScript bundle for a reframed clip.

    Steps:
    1. Verify snippet exists (already audio-synced, 500ms delay baked in).
    2. Compute Position and Scale keyframes for every frame:
         s = out_h / crop_h(t)
         Position = (out_w/2 - (cx - W/2)*s, out_h/2 - (cy - H/2)*s)
         Scale = s * 100
    3. Simplify keyframes with RDP to roughly 5-10 per second.
    4. Write .jsx with data embedded as var DATA = {...}; (no JSON.parse).
    5. Write .srt and _markers.csv.
    6. Validate JS syntax with node --check.
    """
    snip_path = Path(snippet_path).expanduser().resolve()
    if not snip_path.is_file():
        raise FileNotFoundError(f"Audio-synced snippet not found at: {snip_path}")

    out_folder = Path(out_dir).expanduser().resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    base_slug = snip_path.stem
    if base_slug.startswith("snippet_"):
        base_slug = base_slug[len("snippet_"):]
    c_name = comp_name or f"Reel_{base_slug}"

    fps = float(plan.fps)
    num_frames = int(plan.num_frames)
    duration = float(num_frames / fps) if fps > 0 else 0.0
    out_w = int(plan.out_canvas_w)
    out_h = int(plan.out_canvas_h)
    W = float(plan.width)
    H = float(plan.height)

    # 1. Compute per-frame position and scale with default anchor at layer center (W/2, H/2)
    positions = np.zeros((num_frames, 2), dtype=np.float32)
    scales = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        cw = float(plan.crop_w_arr[i]) if plan.crop_w_arr is not None else float(plan.crop_w)
        ch = float(plan.crop_h_arr[i]) if plan.crop_h_arr is not None else float(plan.crop_h)
        cx = float(plan.crop_x0[i]) + cw / 2.0
        cy = float(plan.crop_y0[i]) + ch / 2.0

        s = out_h / ch
        posX = out_w / 2.0 - (cx - W / 2.0) * s
        posY = out_h / 2.0 - (cy - H / 2.0) * s
        scale_val = s * 100.0

        positions[i, 0] = posX
        positions[i, 1] = posY
        scales[i] = scale_val

    # 2. Keyframe simplification via RDP down to roughly 5-10 per second
    cuts = list(plan.scene_cuts) if plan.scene_cuts else None
    key_indices = simplify_keyframes(
        positions=positions,
        scales=scales,
        fps=fps,
        min_rate=max(3.0, target_rate - 2.0),
        max_rate=min(12.0, target_rate + 3.0),
        cuts=cuts,
    )

    kf_times: List[float] = []
    kf_positions: List[List[float]] = []
    kf_scales: List[List[float]] = []

    for k in key_indices:
        t = round(float(k / fps), 4)
        px = round(float(positions[k, 0]), 2)
        py = round(float(positions[k, 1]), 2)
        sc = round(float(scales[k]), 2)

        kf_times.append(t)
        kf_positions.append([px, py, 0.0])
        kf_scales.append([sc, sc, 100.0])

    # 3. Optional copy snippet to export folder for self-contained package
    dest_snippet = out_folder / snip_path.name
    if copy_snippet and snip_path != dest_snippet:
        try:
            shutil.copy2(snip_path, dest_snippet)
            effective_footage_path = str(dest_snippet)
        except Exception:
            effective_footage_path = str(snip_path)
    else:
        effective_footage_path = str(snip_path)

    # 4. Words formatting
    words_data: List[Dict[str, Any]] = []
    for w in words:
        is_emp = getattr(w, "emphasis", False) or (len(w.word) >= 3 and w.word.isupper()) or w.word.endswith("!")
        words_data.append({
            "text": w.word,
            "start": round(float(w.start), 4),
            "end": round(float(w.end), 4),
            "isEmphasis": bool(is_emp),
        })

    # 5. Build DATA structure
    data_payload: Dict[str, Any] = {
        "compName": c_name,
        "compWidth": out_w,
        "compHeight": out_h,
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "footagePath": effective_footage_path,
        "footageFilename": snip_path.name,
        "captionBottomOffset": caption_bottom_offset,
        "captionFontSize": caption_font_size,
        "captionConfig": {
            "fontSize": caption_font_size,
            "bottomOffset": caption_bottom_offset,
            "fontFamily": "Montserrat-ExtraBold",
            "fillColor": [1.0, 1.0, 0.0],
            "strokeColor": [0.0, 0.0, 0.0],
            "strokeWidth": 4,
        },
        "keyframes": {
            "times": kf_times,
            "positions": kf_positions,
            "scales": kf_scales,
        },
        "words": words_data,
        "stats": {
            "totalFrames": num_frames,
            "keyframeCount": len(kf_times),
            "keyframesPerSecond": round(len(kf_times) / max(0.1, duration), 2),
            "wordCount": len(words),
        }
    }

    # 6. Generate Files
    jsx_path = out_folder / f"{base_slug}_ae_setup.jsx"
    generate_extendscript_jsx(data_payload, jsx_path)

    srt_path = out_folder / f"{base_slug}.srt"
    generate_srt(words, srt_path)

    csv_path = out_folder / f"{base_slug}_markers.csv"
    generate_markers_csv(words, fps, csv_path)

    # 7. Validate JS Syntax
    valid_syntax, syntax_msg = validate_jsx_syntax(jsx_path)
    if not valid_syntax:
        raise SyntaxError(f"Generated JSX failed syntax validation: {syntax_msg}")

    print(f"\n[export_ae] Exported After Effects Bundle -> {out_folder}")
    print(f"  • ExtendScript JSX: {jsx_path.name} ({len(kf_times)} keyframes, {data_payload['stats']['keyframesPerSecond']} kf/s)")
    print(f"  • Subtitles SRT   : {srt_path.name} ({len(words)} words)")
    print(f"  • Timeline Markers: {csv_path.name}")
    print(f"  • Synced Footage  : {Path(effective_footage_path).name}")
    print(f"  • JS Syntax Check : {syntax_msg}")

    return {
        "jsx": jsx_path,
        "srt": srt_path,
        "csv": csv_path,
        "snippet": Path(effective_footage_path),
    }
