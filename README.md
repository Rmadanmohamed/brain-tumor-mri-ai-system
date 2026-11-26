---
title: NeuroScope MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# Brain Tumor MRI AI System

[🚀 **Live Demo — Open NeuroScope**](https://rmadanmohamed.github.io/brain-tumor-mri-ai-system/)

## Overview
Brain Tumor MRI AI System is a medical computer vision portfolio project for brain MRI classification experiments, API inference notebooks, segmentation experiments, and a same origin web workspace with a preserved Gradio demo.

> This project is for educational and portfolio demonstration only. It is not a medical diagnosis tool and should not be used for clinical decisions.

## Problem
Brain MRI research workflows often live only in notebooks. This repository keeps the original notebooks while adding a clearer demo entrypoint and honest documentation about model, dataset, and validation limits.

## Features
- MRI image classification workflow
- Responsive scan workspace in `web/`, served by `app.py` and `web_server.py`
- FastAPI inference prototype retained in notebook form
- Segmentation experiment notebook
- Saved Keras model artifact at `source/Image_Detection_Model.h5`
- Clear educational-use disclaimer

## Tech Stack
- Python
- TensorFlow / Keras
- Gradio
- NumPy
- Pillow
- OpenCV
- FastAPI
- PyTorch
- Jupyter Notebook

## Live Demo

**Public application:** [https://rmadanmohamed.github.io/brain-tumor-mri-ai-system/](https://rmadanmohamed.github.io/brain-tumor-mri-ai-system/)

The public GitHub Pages demo runs the converted original trained model in the browser using ONNX Runtime Web. Uploaded images are processed locally in the visitor's browser. The original Python application remains available for local use at `http://127.0.0.1:7860`.

## Medical Disclaimer
This project is for educational and portfolio demonstration only. It is not a medical diagnosis tool and should not be used for clinical decisions. The repository does not include independently verified clinical validation results.

## Dataset
- MRI dataset folders are referenced in the notebooks.
- The full training dataset is not bundled as a clean reproducible dataset package.
- Any public deployment should use only non-sensitive sample images.

## Models / Algorithms
- Saved Keras classifier: `source/Image_Detection_Model.h5`
- MobileNet / CNN-style classification workflow in notebooks
- Segmentation experiment workflow in notebook form

## Results
The notebooks include classification reports, confusion-matrix code, training accuracy tracking, and segmentation metric references. No single final production metric was independently verified across all files, so this README does not claim a final accuracy score.

## Notebook
- `notebooks/brain-tumor-mri-gradio-ui.ipynb`
- `notebooks/brain-tumor-mri-segmentation.ipynb`
- `source/Brain_Tumor_MRI_System.ipynb`
- `source/api.ipynb`

## How to Run Locally

Use 64-bit CPython 3.12 on Windows. The verified interpreter was 3.12.4. From PowerShell, create an environment if needed and invoke that environment explicitly.

```powershell
Set-Location 'D:\projects\brain-tumor-mri-ai-system'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe app.py
```

For an existing installation, run the final command. Use Ctrl+C in the launch terminal to stop the server. If port 7860 is occupied, identify the owner with `netstat -ano | findstr :7860`. The optional `GRADIO_SERVER_PORT` environment variable selects another local port.

Upload a non-sensitive raster image such as JPEG or PNG. The classifier converts it to RGB, resizes it to 224 by 224 with nearest-neighbor interpolation, and rescales float32 pixels by 1/255, following the training generator. The four softmax positions are glioma, meningioma, no tumor, and pituitary. Unrelated images also receive probabilities.

The app reads `source/Image_Detection_Model.h5` relative to the module. Keras 3.8.0 wrote the nested MobileNet endpoints as singleton lists. The loader restores the scalar endpoint references in memory before building the stored graph and reading its weights. The original HDF5 file is the only trained artifact needed. If initialization fails, consult the local server traceback, check the model path and dependencies, then restart. The prediction control is disabled on initialization failure.

Run the regression suite in a second PowerShell window from the project root.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\validate_demo.py
```

The second command requires a running server and local JPEGs under `dataset/Testing/<class>` or an explicit `--sample PATH`. It compares direct and served scores and exercises failed uploads. To reproduce the historical failure with the pinned reader, run `tools\diagnose_model.py` using the same interpreter. Its final native HDF5 load intentionally raises the original exception.

## Project Structure

```text
brain-tumor-mri-ai-system/
  app.py
  requirements.txt
  README.md
  notebooks/
  presentation/
  research/
  source/
    Image_Detection_Model.h5
    Brain_Tumor_MRI_System.ipynb
    api.ipynb
```

## Recommended Deployment

Local validation covers the web API and preserved Gradio classification entry point. The historical notebooks retain their original dependencies and execution assumptions.

## Limitations
- Not validated for clinical use.
- Full dataset packaging and reproducible train/test split are not included.
- Model performance must be rerun and documented before any accuracy claim is made.
- The historical Gradio notebook reverses the final two display labels. Use the verified mapping in the local app. The API and segmentation notebooks are historical experiments.
- Free deployment may be limited by TensorFlow package size and memory.

## Author
Ramadan Mohamed Hassan

- GitHub: https://github.com/Rmadanmohamed
- LinkedIn: https://www.linkedin.com/in/ramadan-mohamed-89085a366/
- Portfolio: https://ramadan-ai-portfolio.vercel.app


## Web workspace and Windows launch

Double click START_APP.bat. It checks the project environment, waits for the model, and opens http://127.0.0.1:7860/. An existing healthy instance is reused. The original Gradio interface remains at /classic locally.

The HTML, CSS, and JavaScript workspace is served from the same Python process as /api/predict. PNG or JPEG uploads are decoded in memory. Requests are limited to 10 MiB and 16 million pixels. The page shows the four actual model probabilities. Opening web/index.html directly redirects to the running local service.

## Deployment preparation

The Dockerfile configures a Hugging Face Docker Space on port 7860 using Python 3.12. It installs requirements-space.txt and copies only the application, web assets, and original HDF5 model. The public container disables /classic to avoid its upload cache. The .dockerignore excludes local datasets, archives, logs, research files, and virtual environments. Paths are portable and MRI_HOST binds the public container.

GitHub Pages hosts a separate static browser deployment on the `browser-deployment` branch. It runs a converted copy of the original trained model with ONNX Runtime Web; the original Python app, Docker configuration, and HDF5 model remain in the main branch for independent local execution.

## Motion experience

The frontend uses CSS and browser animation APIs for the entrance, hero image study, scan accents, navigation, scroll progress, process drawing, and results. The preview scanner runs only while the actual inference request is active. Probability bars and counts animate only toward the four returned model values. Reduced motion preferences disable decorative movement and show final values immediately. The illustration is a decorative rendering and is labeled accordingly.


## Technical reference and reproducibility

### Verified dependency pins

| Runtime or library | Version |
|---|---|
| Python (Docker base) | 3.12-slim |
| TensorFlow | 2.21.0 |
| Keras | 3.14.1 |
| NumPy | 2.4.6 |
| h5py | 3.14.0 |
| Pillow | 12.2.0 |
| FastAPI | 0.136.3 |
| Uvicorn | 0.49.0 |
| Gradio | 6.17.3 |

See requirements.txt and requirements-space.txt for the full pinned environments. The original HDF5 model is bundled; CPU inference is the intended baseline. Historical training and segmentation notebooks use additional dependencies and datasets not fully pinned by the application environment.

### Local setup from a fresh clone (Linux/macOS)

```bash
git clone https://github.com/Rmadanmohamed/brain-tumor-mri-ai-system.git
cd brain-tumor-mri-ai-system
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows PowerShell use `py -3.12 -m venv .venv` and `.\\.venv\\Scripts\\Activate.ps1` before installing. The application serves the main interface at http://127.0.0.1:7860 and the optional Gradio interface at /classic. Platform-specific dependency installation has not been retested in this documentation pass.

### Configuration

| Variable | Required | Default / purpose |
|---|---|---|
| MRI_HOST | No | 127.0.0.1; server bind address |
| GRADIO_SERVER_PORT | No | 7860; HTTP port |
| MRI_CLASSIC | No | 1; set 0 to disable the classic Gradio interface |

No external API credentials are required by the local inference application. The Docker configuration binds 0.0.0.0 and disables the classic interface.

### API usage

```bash
curl http://127.0.0.1:7860/api/health
curl -X POST http://127.0.0.1:7860/api/predict \\
  -H 'Content-Type: image/jpeg' --data-binary @YOUR_MRI_IMAGE.jpg
```

Replace the placeholder with a local JPEG; PNG is also accepted with its corresponding media type. Health returns app and ready fields. Prediction returns classification and scores (four class scores). The body is raw image bytes, not multipart JSON. Invalid content, oversized input, unavailable model and failed inference are handled by HTTP error responses. Limits: 10 MiB and 16 million pixels.

### Model and dataset evidence

The application loads the saved Keras HDF5 classifier, applies RGB conversion, nearest-neighbor resizing to 224 × 224 and normalization by 255, then maps outputs to glioma, meningioma, no tumor and pituitary. The training notebook uses MobileNet, Adam (learning rate 0.0001) and categorical cross-entropy; the original dataset ZIP is not bundled. The segmentation notebook is a separate U-Net experiment using combined BCE/Dice loss and IoU evaluation; it is not the deployed classifier. No independently verified test-set metric was established from saved notebook outputs during this audit. Dataset provenance, licensing and complete split reproducibility remain outstanding.

### Validation scope

```bash
python -m unittest discover -s tests -v
python tools/validate_demo.py --url http://127.0.0.1:7860/classic
```

The classifier suite checks the saved model, preprocessing, output dimensions, invalid inputs and repeatability; the sample-image test is conditional on local dataset availability. The demo-validation script requires the running classic interface and local test images. The tests were inspected, not executed during this documentation audit. The browser-deployment branch contains the ONNX artifact, browser runtime and static UI; the public GitHub Pages URL could not be independently loaded during this audit, so live availability is not asserted.

## Author

Ramadan Mohamed Hassan — AI Engineer | AI Researcher | AI Automation | LLM & RAG Systems | Machine Learning

[Portfolio](https://ramadan-ai-portfolio.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/ramadan-mohamed-89085a366/) · [GitHub](https://github.com/Rmadanmohamed) · [Email](mailto:ramadanmohameddd3@gmail.com)
