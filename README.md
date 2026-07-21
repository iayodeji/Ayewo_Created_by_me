Ayewo Backend
Ayewo is an AI-powered malaria blood smear screening backend designed for resource-limited Nigerian diagnostic labs. It leverages deep learning (MobileNetV2) to rapidly classify cell images and generate PDF diagnostic reports.
Tech Stack
Runtime: Python 3.10+ managed via `uv`
API Framework: FastAPI & Uvicorn
Deep Learning: PyTorch + Torchvision (MobileNetV2 architecture)
Explainable AI: `pytorch-grad-cam` (for visual heatmaps of infection markers)
Imaging & Reports: Pillow & ReportLab
---
Complete Step-by-Step Setup
Follow these steps in order to get the backend fully operational.


Step 1: Initialize Project & Install Dependencies
Instead of dealing with manual virtual environments and slow `pip` installs, `uv` will sync your entire environment instantly.
Open your terminal in the root project folder (`Datican_Ayewo`) and run:
```bash
# This reads your pyproject.toml / requirements, creates a .venv, and installs everything
uv sync
```
Step 2: Download the Malaria Dataset
The project trains on the NIH cell image dataset from Kaggle.
Run your download script to fetch and extract it to your local cache:
```bash
uv run Ayewo/malaria_dataset.py
```
Note the path printed at the end of this script execution (typically `C:\Users\HP\.cache\kagglehub\...`). You will need it for the next step.

Step 3: Train the Model
Train the MobileNetV2 classifier for at least 5 epochs. Make sure to point the `--data-dir` to the exact path where your dataset was extracted (and append `/cell_images` if it contains that nested folder).
```bash
uv run Ayewo/train.py \
  --data-dir "C:\Users\HP\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\cell_images" \
  --epochs 5 \
  --output malaria_model.pth
```
This will save your trained weights as `malaria_model.pth` in your project folder.
Step 4: Run the FastAPI Application
Once training is complete and your `malaria_model.pth` is ready, launch the server using `uv run`:
```bash
uv run uvicorn Ayewo.main:app --host 0.0.0.0 --port 8000
``` 
The backend is now live at `http://127.0.0.1:8000`!
---
API Testing Examples
You can test the running API endpoints using `curl` from another terminal window.
1. Health Check
Verify the API is up and running.
```bash
curl "http://127.0.0.1:8000/health"
```
2. Single Image Screening
Upload a single blood smear image for instant prediction.
```bash
curl -X POST "http://127.0.0.1:8000/predict/single" \
  -F "file=@/absolute/path/to/image.png"
```
3. Batch Screening
Upload multiple images simultaneously for high-throughput screening.
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -F "files=@/absolute/path/to/image1.png" \
  -F "files=@/absolute/path/to/image2.png"
```
4. Download Diagnostic PDF Report
Retrieve a generated PDF summary report for a specific batch screening session.
```bash
curl -L "http://127.0.0.1:8000/report/<batch_id>" -o batch_report.pdf
```

For high-throughput screening, skip the optional Grad-CAM explanation image:

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch?include_gradcam=false" \
  -F "files=@/absolute/path/to/image1.png" \
  -F "files=@/absolute/path/to/image2.png"
```
---
System Operational Notes
Frontend Integration: CORS is explicitly enabled for all origins (`*`) to facilitate easy local testing and smooth frontend connection.
Quality Assurance: Any diagnostic prediction carrying less than 75% confidence is automatically flagged with `low_confidence: true` to prompt manual technician review.
Data Architecture: Built completely lightweight for speed—no authentication, database overhead, or user session state management required.

Serving notes: the model is loaded once at application startup. Each upload is capped at 10 MB, batches are capped at 50 images, and the latest 500 batch summaries remain in memory for report downloads.
