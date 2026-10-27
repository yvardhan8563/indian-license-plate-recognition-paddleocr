# Indian License Plate Detection & Recognition

### Lynkit — AI/ML Computer Vision Assessment

## Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** pipeline using **YOLO11n, PaddleOCR PP-OCRv4, and PP-OCRv5**.

The system detects license plates from vehicle images, extracts registration numbers, and compares the accuracy and confidence of both OCR models.

## Technologies

Python · PaddleOCR · PaddlePaddle · YOLO11n · OpenCV · PyTorch · Google Colab

## Project Workflow

```text
Vehicle Image
     ↓
YOLO11n Plate Detection
     ↓
License Plate Cropping
     ↓
PP-OCRv4 / PP-OCRv5
     ↓
Recognized Text + Confidence
     ↓
Accuracy & Error Analysis
```

## Dataset & Training

- 1,695 annotated license plate crops
- Training images: 1,365
- Validation images: 330
- OCR fine-tuning: 20 epochs
- Custom dictionary: A–Z and 0–9
- Vehicle images and XML annotations used for detection training

## Model Performance

| Metric | PP-OCRv4 | PP-OCRv5 |
|---|---:|---:|
| Validation accuracy | 56.67% | **82.12%** |
| Normalized edit similarity | 92.47% | **96.55%** |
| Correct predictions (330 images) | 187 | **271** |
| Full-image demonstration accuracy (20 images) | 45% | **90%** |

YOLO detected license plates in all 20 demonstration images.

*Note: The 20-image demonstration is not an independent end-to-end benchmark because the OCR and detector splits were created separately.*

## Repository Files

- `PPOCR_License_Plate_Training_Submission.ipynb` — Data preparation, model fine-tuning, and validation.
- `PPOCR_License_Plate_Inference_Submission.ipynb` — Plate detection, OCR inference, confidence analysis, batch processing, and error visualization.

## How to Run

1. Open the notebooks in Google Colab with GPU enabled.
2. Install the dependencies using the provided cells.
3. Provide vehicle images, XML annotations, and required model checkpoints.
4. Execute the training or inference workflow.

The trained checkpoints are stored separately. Recorded outputs are included, but fresh-runtime execution has not been independently verified.

## Conclusion

**PP-OCRv5 outperformed PP-OCRv4**, achieving 82.12% validation accuracy compared with 56.67%. The project demonstrates license plate detection, recognition, confidence scoring, and model comparison.

---

**Lynkit | AI/ML Computer Vision Skill Assessment**
