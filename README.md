# Pneumonia Detection on Chest X-rays

Udacity AI for Healthcare Nanodegree – Course 2 Project

## Overview
Binary classification model for detecting pneumonia on frontal chest X-ray images. Built with VGG16 transfer learning and prepared for FDA 510(k) submission documentation.

## Files
| File | Description |
|------|-------------|
| `EDA.ipynb` | Exploratory data analysis on NIH Chest X-ray dataset |
| `Build and train model.ipynb` | Model training with configuration comparison |
| `Inference.ipynb` | DICOM validation and inference pipeline |
| `FDA_Submission_Final.pdf` | FDA submission documentation |
| `my_model.json` | Model architecture |
| `my_model_weights.h5` | Trained model weights |

## Model
- **Architecture:** VGG16 (block5 fine-tuned) + Dense head
- **Input:** 224×224 RGB chest X-rays
- **Threshold:** 0.523
- **Validation AUC:** 0.683
- **Validation F1:** 0.416

## Training Configurations Compared
| Config | LR | Epochs | Best Val Loss | AUC | F1 |
|--------|----|--------|---------------|-----|----|
| Original | 1e-4 | 20 | 0.494 | 0.684 | 0.424 |
| Config A | 1e-4 | 20 | 0.455 | 0.683 | 0.414 |
| Config C | 1e-4 | ~30 | 0.447 | 0.683 | 0.416 |

## Inference
The inference notebook:
1. Validates DICOM headers (Modality=DX, BodyPart=CHEST, View=AP/PA)
2. Preprocesses images (resize 224×224, RGB, [0,1])
3. Runs model prediction
4. Applies threshold for binary output

## Requirements
- Python 3.7+
- TensorFlow/Keras
- pydicom
- scikit-image
- scikit-learn
- pandas, numpy

## Author
Sandip Dey
