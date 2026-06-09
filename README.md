# Brain Tumor MRI AI System

## Overview
Brain Tumor MRI AI System is a medical computer vision portfolio project for brain MRI classification experiments, API inference notebooks, segmentation experiments, and a Gradio demo app.

> This project is for educational and portfolio demonstration only. It is not a medical diagnosis tool and should not be used for clinical decisions.

## Problem
Brain MRI research workflows often live only in notebooks. This repository keeps the original notebooks while adding a clearer demo entrypoint and honest documentation about model, dataset, and validation limits.

## Features
- MRI image classification workflow
- Gradio demo app in `app.py`
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
Live demo: Not available yet.

Reason: The Gradio app has been prepared for Hugging Face Spaces, but local verification found that the bundled `source/Image_Detection_Model.h5` model does not load successfully with the current TensorFlow/Keras environment. The app handles this safely by showing a model-unavailable message instead of producing fake predictions.

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

```bash
git clone https://github.com/Rmadanmohamed/brain-tumor-mri-ai-system.git
cd brain-tumor-mri-ai-system
python -m venv .venv
pip install -r requirements.txt
python app.py
```

The app expects the model file at:

```text
source/Image_Detection_Model.h5
```

If TensorFlow or the model cannot be loaded, the app will show a clear error instead of silently producing a fake prediction.

Current local model-load result:

```text
Could not load model: Layer "dense" expects 1 input(s), but it received 2 input tensors.
```

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
Hugging Face Spaces with Gradio is the best target for this project after the model artifact is repaired or re-exported. Until then, keep the project as GitHub + notebook documentation only and do not publish a prediction demo.

## Limitations
- Not validated for clinical use.
- Full dataset packaging and reproducible train/test split are not included.
- Model performance must be rerun and documented before any accuracy claim is made.
- The current saved `.h5` artifact fails to load in the tested TensorFlow/Keras environment and should be re-exported from the original training notebook.
- Free deployment may be limited by TensorFlow package size and memory.

## Author
Ramadan Mohamed Hassan

- GitHub: https://github.com/Rmadanmohamed
- LinkedIn: https://www.linkedin.com/in/ramadan-mohamed-657686365
- Portfolio: https://ramadan-ai-portfolio.vercel.app
