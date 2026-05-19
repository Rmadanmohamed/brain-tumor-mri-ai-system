# Brain Tumor MRI AI System

## Overview
A medical computer vision project for brain tumor MRI classification, API inference, segmentation experiments, and simple UI prototypes.

## Problem Statement
Brain MRI review is time-sensitive and requires reliable image classification workflows that can support clinical decision-making research.

## Features
- MRI image classification workflow
- FastAPI and Gradio inference prototypes
- Segmentation notebook using deep learning
- Pretrained model artifact included when available

## Tech Stack
- Python
- TensorFlow
- Keras
- MobileNet
- PyTorch
- FastAPI
- Gradio
- OpenCV

## Dataset
- Braintumer Dataset.zip
- MRI image folders referenced in notebooks

## Models / Algorithms
- MobileNet transfer learning
- CNN classifier
- U-Net / ResNet segmentation experiment

## Results
The notebooks include classification reports, confusion-matrix code, training accuracy tracking, and Dice/IoU evaluation references. No single final production metric was verified across all files.

## Project Structure
- notebooks/
  - brain-tumor-mri-gradio-ui.ipynb
  - brain-tumor-mri-segmentation.ipynb
- presentation/
  - Presentation op (1).pptx
- research/
  - Brain Tumor Classification Using Deep Learning.pdf
  - jimaging-10-00275 (1).pdf
- source/
  - api.ipynb
  - Brain Tumer Classification Presentation.pdf
  - Brain_Tumor_MRI_System.ipynb
  - Image_Detection_Model.h5

## How to Run
- `pip install -r requirements.txt`
- `jupyter notebook notebooks/brain-tumor-mri-classification.ipynb`
- `uvicorn api:app --reload`

## Screenshots / Demo
Screenshots or demo media can be added after deployment.

## Future Improvements
- Add a reproducible train/test split manifest
- Convert notebooks into a tested package
- Add model cards and validation caveats

## Author
Ramadan Mohamed Hassan
