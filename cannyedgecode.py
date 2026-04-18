# CELL 1 — Install dependencies
!pip install opencv-python-headless gradio gradio_client openai -q
!apt-get install -y ffmpeg -q
print("✅ Done")

# CELL 2 — Imports & config
import cv2
import numpy as np
import subprocess
import os
import base64
import json

from gradio_client import Client, handle_file
from openai import OpenAI
from IPython.display import display, Image as IPImage
from google.colab import files

# ── CONFIG — edit these ───────────────────────────────────────────────────────
VIDEO_NAME        = "lecture_video"        # no extension
VIDEO_EXT         = ".mp4"
CLIPS_OUTPUT_DIR  = f"/content/{VIDEO_NAME}_clips"

HF_SPACE          = "YOUR_HF_USERNAME/board-wipe-recovery"   # update after HF deploy

DATABRICKS_TOKEN  = "dapi4698106c12488aaf247f45a7834a1aed"
GATEWAY_BASE_URL  = "https://7474660314620622.ai-gateway.cloud.databricks.com/mlflow/v1"
VLM_MODEL_NAME    = "lecture-description"

os.makedirs(CLIPS_OUTPUT_DIR, exist_ok=True)
print("✅ Config ready")

# CELL 3 — Upload video from your machine into Colab
uploaded = files.upload()   # opens file picker

# grab the filename of whatever was uploaded
uploaded_filename = list(uploaded.keys())[0]
VIDEO_PATH = f"/content/{uploaded_filename}"

# rename to match VIDEO_NAME if needed (keeps downstream cells clean)
canonical_path = f"/content/{VIDEO_NAME}{VIDEO_EXT}"
if VIDEO_PATH != canonical_path:
    os.rename(VIDEO_PATH, canonical_path)
    VIDEO_PATH = canonical_path

print(f"✅ Video ready at: {VIDEO_PATH}")

# Quick sanity check
cap = cv2.VideoCapture(VIDEO_PATH)
fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration     = total_frames / fps
cap.release()
print(f"   FPS: {fps:.1f}  |  Frames: {total_frames}  |  Duration: {duration:.1f}s  ({duration/60:.1f} min)")

# CELL 4 — Canny edge density gradient analysis
# Finds inflection points where gradient was negative (board being wiped)
# and turns positive (prof starting to write again) = local minima of edge density

def detect_wipe_recovery_points(
    video_path,
    sample_interval_sec=1.0,
    canny_low=50,
    canny_high=150,
    min_drop_duration_sec=3.0,
    min_density_drop=0.005,
    verbose=True
):
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur    = total_frames / fps
    step         = int(fps * sample_interval_sec)

    if fps == 0 or step == 0:
        raise RuntimeError("Could not read video FPS")

    # ── 1. sample frames → edge density at each timestamp ───────────────────
    densities = []
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(gray, canny_low, canny_high)
        density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        densities.append((frame_idx / fps, density))
        frame_idx += step

    cap.release()

    if len(densities) < 3:
        raise RuntimeError("Not enough frames sampled — video too short or interval too large")

    # ── 2. smooth density signal (3-sample rolling average) ─────────────────
    raw   = [d for (_, d) in densities]
    smooth = [
        sum(raw[max(0, i-1) : i+2]) / len(raw[max(0, i-1) : i+2])
        for i in range(len(raw))
    ]

    # ── 3. compute per-step gradients ────────────────────────────────────────
    gradients = [smooth[i] - smooth[i-1] for i in range(1, len(smooth))]

    # ── 4. find local minima (inflection: falling → rising) ─────────────────
    recovery_timestamps = []
    in_fall      = False
    fall_start_i = None
    fall_start_v = None

    for i, grad in enumerate(gradients):
        if not in_fall:
            if grad < 0:
                in_fall      = True
                fall_start_i = i
                fall_start_v = smooth[i]
        else:
            if grad >= 0:
                in_fall = False
                inflection_ts  = densities[i][0]
                inflection_val = smooth[i]
                drop_amount    = fall_start_v - inflection_val
                drop_duration  = inflection_ts - densities[fall_start_i][0]

                if drop_amount >= min_density_drop and drop_duration >= min_drop_duration_sec:
                    mm = int(inflection_ts // 60)
                    ss = int(inflection_ts % 60)
                    ts_str = f"{mm:02d}:{ss:02d}"
                    recovery_timestamps.append(ts_str)
                    if verbose:
                        print(f"  Recovery at {ts_str}  |  drop: {drop_amount:.4f}  duration: {drop_duration:.1f}s")

    if verbose:
        print(f"\n✅ {len(recovery_timestamps)} recovery points found in {total_dur:.1f}s of video")

    return recovery_timestamps, total_dur


print("Running Canny wipe detection...")
wipe_timestamps_str, total_duration = detect_wipe_recovery_points(
    VIDEO_PATH,
    sample_interval_sec  = 1.0,
    canny_low            = 250,
    canny_high           = 255,
    min_drop_duration_sec= 3.0,
    min_density_drop     = 0.005,
)

print("\nRecovery timestamps:", wipe_timestamps_str)