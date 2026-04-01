# Deepfake Detector

Deepfake Detector is a research-oriented FastAPI app and CLI for analyzing videos with two detection pipelines:

- `MesoNet` for face-level artifact detection
- `Temp-D3` for temporal anomaly detection across frames

The repository is set up to be open-source friendly: application code is included, but pretrained weights and datasets are not redistributed.

## What Is In This Repo

This repository includes:

- application code in `main.py`, `run_models.py`, `templates/`, and `static/`
- vendored or adapted research code in `MesoNet/` and `temp-d3/`
- small sample assets and supporting project files

This repository intentionally does not include:

- pretrained `MesoNet` weights
- datasets
- downloaded videos
- cached external model downloads

If you use local weights for experimentation, keep them untracked under `MesoNet/weights/`.

## Licensing

- The original code in this repository is licensed under the MIT License. See [LICENSE](LICENSE).
- Vendored research code and other third-party material are not automatically relicensed under MIT. See [THIRD_PARTY.md](THIRD_PARTY.md).
- Only add or redistribute weights, datasets, or source media if you have clear rights to do so.

## How It Works

For a YouTube URL submitted through the web app:

1. `POST /api/analyze` validates the URL and creates an in-memory job.
2. `yt-dlp` downloads the source video into `downloads/`.
3. `run_models.py` runs `MesoNet` when compatible local weights are available and runs `Temp-D3`.
4. `main.py` parses the model outputs, applies a sigmoid to the raw `Temp-D3` score, and combines available scores.
5. The backend returns a final `real` or `fake` verdict and deletes the temporary video when cleanup succeeds.

`MesoNet` is optional in practice. If no local weights are installed, the system falls back to `Temp-D3` only.

### Model Outputs

| Model | Purpose | Output |
| --- | --- | --- |
| `MesoNet (Meso4)` | Detects face-level forgery artifacts | Score from `0.0` to `1.0`, where higher is more fake-like |
| `Temp-D3 (XCLIP-16)` | Detects temporal anomalies in video frames | Raw anomaly score, where higher is more suspicious |

### Fusion Logic

```text
temp_d3_score = sigmoid(temp_d3_raw_score)
combined_score = mean([temp_d3_score, mesonet_score])    # when MesoNet produced a score
combined_score = temp_d3_score                           # when MesoNet is unavailable or unusable
verdict = "fake" if combined_score >= 0.5 else "real"
```

## Repository Structure

```text
Deepfake_Detector/
├── main.py
├── run_models.py
├── requirements_web.txt
├── templates/
├── static/
├── MesoNet/
├── temp-d3/
├── LICENSE
├── THIRD_PARTY.md
```

## Requirements

Recommended baseline:

- Python `3.11`
- `pip`
- `ffmpeg`
- `cmake` and a working compiler toolchain for `dlib` and `face_recognition`
- internet access on first `Temp-D3` run so the encoder can be cached locally

Optional:

- a CUDA-capable GPU for faster inference
- local `MesoNet` weights if you want the face-based detector enabled

### System Dependencies

macOS:

```bash
brew install ffmpeg cmake
```

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg cmake build-essential
```

Some environments may also need OpenCV runtime packages such as `libgl1`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ishaan2011/Deepfake-detector.git
cd Deepfake-detector
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install the web app dependencies

```bash
pip install -r requirements_web.txt
```

### 4. Install Temp-D3 dependencies

If you need a specific CUDA build of PyTorch, install the matching `torch` and `torchvision` packages first, then run:

```bash
pip install -r temp-d3/requirements.txt
```

### 5. Install optional MesoNet runtime dependencies

```bash
pip install tensorflow keras face_recognition scipy imageio imageio-ffmpeg
```

### 6. Add optional local assets

- `Temp-D3` may download pretrained encoder weights on first run.
- `downloads/` is created automatically by `main.py`.
- If you have rights to use compatible `MesoNet` weights, place them under `MesoNet/weights/`.

For offline runs, pre-cache the model referenced by `temp-d3/models/D3_model.py`, which currently defaults to `microsoft/xclip-base-patch16`.

## Usage

### Run the web app

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) and submit a YouTube URL.

### Run the CLI on a local video

```bash
python run_models.py /path/to/video.mp4 --verbose-status
```

Example output:

```text
MesoNet not used (No faces found or error)
Temp-D3 Score: 3.4512 (Higher value ~ more likely Fake/Anomaly)
```

If local `MesoNet` weights are installed, the CLI will print a `MesoNet Score` as well.

## API

### `GET /`

Serves the browser UI.

### `POST /api/analyze`

Starts a YouTube analysis request.

Request:

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=..."
}
```

Response:

```json
{
  "job_id": "abc123",
  "status_endpoint": "/api/analyze/abc123"
}
```

### `GET /api/analyze/{job_id}`

Returns the current job snapshot, including status, logs, progress, and the final result when complete.

Example completed response:

```json
{
  "job_id": "abc123",
  "youtube_url": "https://www.youtube.com/watch?v=...",
  "status": "completed",
  "phase": "Completed",
  "download_percent": 100.0,
  "result": {
    "video_file": "abcd1234.mp4",
    "mesonet_score": 0.82,
    "temp_d3_raw_score": 3.45,
    "temp_d3_score": 0.97,
    "overall_verdict": "fake",
    "processing_seconds": 42.1
  },
  "error": null
}
```

Observed backend states:

- `queued`
- `started`
- `downloading`
- `running_models`
- `running_mesonet`
- `running_temp_d3`
- `completed`
- `failed`

## Project Notes

- Jobs are stored in memory and processed in background threads.
- Request history is lost on server restart.
- Downloaded videos are stored temporarily in `downloads/` and deleted when cleanup succeeds.

## Troubleshooting

### `yt-dlp is not installed`

```bash
pip install -r requirements_web.txt
```

### `ffmpeg` not found

Install `ffmpeg` and make sure it is available on your `PATH`.

### `face_recognition` or `dlib` fails to install

Install `cmake` and build tools first, then retry. Python `3.11` is usually the smoothest option.

### First `Temp-D3` run is slow

The first run may download encoder weights from Hugging Face. Later runs should use the local cache.

### `MesoNet` returns no score

That usually means one of two things:

- no compatible local `MesoNet` weights are installed
- the pipeline could not extract enough usable faces from the video

In either case, the backend falls back to the normalized `Temp-D3` score alone.

## Limitations

- The browser workflow accepts YouTube URLs only.
- Jobs are stored in memory and run in background threads.
- Request history is lost on server restart.
- There is no persistent database, auth layer, rate limiting, or worker queue.
- Accuracy depends heavily on compression, lighting, motion, and domain shift.
- `MesoNet` is face-dependent and may fail on videos without stable detectable faces.
- This repository mixes app code with vendored research components.
- There is currently no automated test suite or CI pipeline in the repository.

This project should not be treated as forensic proof or as a sole decision-making system.

## Responsible Use

- Use results as a signal, not final proof.
- Verify suspicious content with multiple methods and human review.
- Do not use this tool as the sole basis for legal, safety-critical, or reputational decisions.
- Respect the licenses and usage terms of upstream models, weights, datasets, and source media.

## Contributing

Focused pull requests are easiest to review. If you change model orchestration, keep the API flow and CLI behavior aligned so the web app and local runner stay consistent.

## Acknowledgements

This project builds on or incorporates ideas from:

- `MesoNet`: Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection Network"
- `Temp-D3`: Zheng et al., "D3: Training-Free AI-Generated Video Detection Using Second-Order Features"
