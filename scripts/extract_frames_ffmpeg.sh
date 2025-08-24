#!/bin/bash

# Directory with input videos
VIDEO_DIR="path/to/input_videos"
# Output directory for extracted frames
OUTPUT_DIR="path/to/output_frames"

mkdir -p "$OUTPUT_DIR"

for video in "$VIDEO_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    out_dir="$OUTPUT_DIR/$filename"
    mkdir -p "$out_dir"

    echo "Extracting frames from $filename..."
    ffmpeg -i "$video" -vf "fps=1" "$out_dir/frame_%04d.jpg"
done