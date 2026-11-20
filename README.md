
# satellite-land_cover-segmentation
Semantic segmentation of aerial and satellite imagery for land-cover mapping using the LandCover.ai dataset, comparing Baseline models, U-Net, DeepLabV3+, and SegFormer architectures.

# Dataset Preprocessing Notebook

Original Kaggle Notebook:

[Open Dataset ](https://www.kaggle.com/code/shazahanybakr/dataset)

This notebook contains the original dataset preprocessing workflow.


## Model Checkpoints

- **Minimal CNN (Baseline):** [`minimal_cnn_best.keras`](checkpoints/minimal_cnn_best.keras)
- **SegFormer-B0 (Modern Model):** [Download Instructions](checkpoints/segformer_b0.md)

Note: The SegFormer-B0 checkpoint is hosted in GitHub Releases due to its file size.
=======

# Satellite Land Cover Segmentation

## Project Overview

An AI-powered Deep Learning application designed
to perform pixel-level segmentation of satellite imagery.

## Land Cover Classes

1. Forest
2. Water
3. Agriculture
4. Buildings
5. Bare Soil
6. Roads

## Technologies

- Python
- Gradio
- PyTorch
- Hugging Face
- NumPy
- Pillow

## Project Structure

Satellite-Land-Cover-Segmentation/
├── app.py
├── inference.py
├── visualization.py
├── requirements.txt
├── README.md
├── assets/
├── models/
├── outputs/
└── venv/


## Current Status

- [x] Gradio dashboard
- [x] Image upload and preview
- [x] Dark and Light modes
- [x] Project structure
- [x] Visualization utilities
- [ ] Model integration
- [ ] Real segmentation predictions
- [ ] Statistics integration
- [ ] Hugging Face deployment

## Run Locally

pip install -r requirements.txt
python app.py


## Project Note

The prediction model will be integrated
after connecting the trained model.
4be1942 (Add satellite land cover demo)
