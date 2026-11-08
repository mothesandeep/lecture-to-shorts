#!/usr/bin/env bash
set -euo pipefail

# scripts/make_sample.sh - Cuts a 30s sample clip for benchmarking and testing.
# Default target: samples/sample.mp4

INPUT="${1:-}"

# Fallback to Desktop lecture video if none provided
if [ -z "$INPUT" ]; then
    if [ -f "/Users/sandeep/Desktop/lecture_hook_25fps.mov" ]; then
        INPUT="/Users/sandeep/Desktop/lecture_hook_25fps.mov"
    elif [ -f "/Users/sandeep/Desktop/lecture_hook_25fps_synced.mov" ]; then
        INPUT="/Users/sandeep/Desktop/lecture_hook_25fps_synced.mov"
    else
        echo "Usage: $0 <input_video_path> [start_time_seconds]" >&2
        exit 1
    fi
fi

START_TIME="${2:-0.0}"
DURATION=30.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLES_DIR="$PROJECT_ROOT/samples"
OUTPUT_FILE="$SAMPLES_DIR/sample.mp4"

mkdir -p "$SAMPLES_DIR"

echo "=========================================================="
echo " CUTTING 30s BENCHMARK SAMPLE"
echo "=========================================================="
echo "• Input Source : $INPUT"
echo "• Start Offset : ${START_TIME}s"
echo "• Duration     : ${DURATION}s"
echo "• Output Target: $OUTPUT_FILE"
echo "=========================================================="

ffmpeg -y -ss "$START_TIME" -i "$INPUT" -t "$DURATION" \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -r 25 -fps_mode cfr \
    -c:a aac -b:a 192k -af "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0" \
    -movflags +faststart "$OUTPUT_FILE"

echo ""
echo ">>> Sample generated successfully: $OUTPUT_FILE"
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUTPUT_FILE"

