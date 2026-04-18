# FAORE : Frame Aware Obstruction Remover and Explainer.

![Architecture Pipeline](WhatsApp%20Image%202026-04-18%20at%2003.14.02.jpeg)

## Prologue: Future Scope and The FAOR Model

*Note: The model implementation detailed below is not yet complete. The final proposed pipeline will incorporate the advanced capabilities outlined here.*

While the standard pipeline handles generally stable video feeds (such as stationary lecture recordings) as a simple use case, the true power of the **FAOR** architecture lies in its ability to process **unstable video footage**, such as handheld recordings or live photos.

To achieve this, FAOR utilizes **matrix transformations** and **ORB keypoints descriptors** to precisely align the board frames prior to fusion. This ensures that the exact context on the board is completely retained, even with significant camera movement.

**Broader Applications:**
* **Context-Aware Object Removal:** This approach serves as a highly accurate alternative to standard AI image erasers. Instead of inpainting over an object with hallucinated, meaningless content, FAOR can leverage multiple aligned frames to computationally eliminate obstructions while revealing the actual, factual background.
* **Traffic Enforcement in India:** This technology has multiple practical use cases in India, where speeding cars sometimes evade traffic fines due to transient obstructions blocking the camera's line of sight. By aligning and fusing adjacent frames, these obstructions can be seamlessly eliminated, revealing the vehicle clearly without compromising the evidentiary context of the scene.

---

## What it does

1. Takes a lecture video as input
2. Detects every moment the professor finishes wiping the board and starts
   writing again — using Canny edge density gradient analysis
3. Cuts the video into individual segments at those boundaries using FFmpeg
4. Sends each clip to a custom HuggingFace model (FAOR) that extracts a clean
   image of the board content
5. Passes each cleaned board image to a VLM hosted on Databricks AI Gateway
   (Gemma 3 12b) and generates a detailed description of the concepts shown

---

## Components

### Canny wipe detection (`hf_space/app.py`)

Hosted as a Gradio Space on HuggingFace. Accepts a video file and returns an
array of `MM:SS` timestamps — one per wipe recovery point.

**Algorithm:**
- Sample one frame every N seconds
- Apply Gaussian blur then Canny edge detection to each frame
- Compute edge pixel density (fraction of pixels that are edges)
- Smooth the density signal with a 3-sample rolling average
- Compute per-step gradients of the smoothed signal
- Detect local minima: where gradient transitions from negative to positive
- Apply quality filters: minimum drop depth and minimum drop duration
- Return the timestamp of each passing local minimum

The local minimum is the optimal cut point — the board is maximally empty at
that moment, so segments start and end at the cleanest possible state.

**Tunable parameters (exposed in Gradio UI):**

| Parameter | Default | Effect |
|---|---|---|
| `sample_interval_sec` | 1.0 | Frame sampling rate — lower = finer but slower |
| `canny_low` | 50 | Lower Canny threshold — raise to reduce noise sensitivity |
| `canny_high` | 150 | Upper Canny threshold |
| `min_drop_duration_sec` | 3.0 | Minimum wipe duration — filters out brief flickers |
| `min_density_drop` | 0.005 | Minimum edge density reduction — filters shallow dips |

---

### FFmpeg segmentation

Uses `subprocess` to call FFmpeg directly. Each segment is named
`video_name_t1-t2.mp4` where `t1` and `t2` are the start and end times in
seconds (with dots replaced by underscores for filesystem safety).

Re-encodes with `libx264` + `aac` to ensure clean keyframe alignment at every
cut boundary.

---

### HuggingFace FAOR model (`imperiusrex/FAOR`)

A custom model that takes a lecture video clip as input and returns a clean,
processed image of the board content. Called via the `gradio_client` API.

---

### Databricks VLM

The cleaned board image is base64-encoded and sent to a Gemma 3 12b endpoint
via the Databricks AI Gateway using an OpenAI-compatible client. Returns a
natural language description of the concepts written on the board.

**Endpoint:** `https://7474660314620622.ai-gateway.cloud.databricks.com/mlflow/v1`  
**Model name:** `lecture-description`

---

## Getting started

### Run in Google Colab (recommended for testing, can also run on DataBricks)
First Make a New Notebook on Colab
1. Open `main.py` in Colab
2. Run Cell 1 to install dependencies (includes FFmpeg)
3. Run Cell 2 and set your config values
4. Run Cell 3 to upload your video
5. Run Cells 4 and 5 — Cell 5 plots the edge density curve so you can
   visually verify that wipe recovery points are landing in the right places
6. Adjust `min_drop_duration_sec` and `min_density_drop` in Cell 4 if needed,
   then re-run
7. Run Cells 6–10 to complete the pipeline
8. Run Cell 11 to download the HuggingFace Space files for deployment

### Deploy the Canny detector to HuggingFace

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Upload `hf_space/app.py`, `hf_space/requirements.txt`, `hf_space/README.md`
4. Update `HF_SPACE` in your config to point to your new Space

### Run in Databricks

1. Create a new Databricks notebook
2. Paste each `databricks_notebook/cell_N_*.py` file as a separate cell in order
3. Update Cell 2 config with your video path, HF Space URL, token, and gateway URL
4. Run all cells sequentially

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python-headless` | Canny edge detection, frame sampling |
| `gradio` / `gradio_client` | HuggingFace Space hosting and API calls |
| `openai` | Databricks AI Gateway client (OpenAI-compatible) |
| `ffmpeg` | Video segmentation (system package) |
| `numpy` | Edge density computation |
| `matplotlib` | Density curve visualisation (Colab only) |

---

## Configuration reference

| Variable | Description |
|---|---|
| `VIDEO_NAME` | Base name for the video and output clips (no extension) |
| `VIDEO_PATH` | Full path to the source video |
| `CLIPS_OUTPUT_DIR` | Directory where segmented clips are saved |
| `HF_SPACE` | HuggingFace Space identifier (`username/space-name`) |
| `DATABRICKS_TOKEN` | Databricks personal access token |
| `GATEWAY_BASE_URL` | Databricks AI Gateway base URL |
| `VLM_MODEL_NAME` | Model endpoint name on the gateway |

> For production use on Databricks, replace the hardcoded token with
> `dbutils.secrets.get(scope="...", key="...")`.

---

## Notes

- FFmpeg is pre-installed on Databricks ML Runtime clusters. If it is missing,
  add `%sh apt-get install -y ffmpeg` before the segmentation cell.
- If your video is stored in a Unity Catalog volume, adjust `VIDEO_PATH` to
  `/Volumes/catalog/schema/volume/filename.mp4` and ensure the cluster has
  volume access.
- The Canny detector works best on blackboard/whiteboard recordings with
  reasonable lighting. Heavy video compression artefacts can slightly elevate
  the baseline edge density — increase `canny_low` in that case.
- Segments shorter than 5 seconds are automatically dropped as likely noise.

---

## License

MIT
