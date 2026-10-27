# Indian Vehicle License Plate Recognition — Lynkit Computer Vision Assessment

Two Google Colab notebooks compare fine-tuned PaddleOCR PP-OCRv4 and PP-OCRv5 on Indian license plate recognition. The inference notebook also demonstrates a YOLO11n plate detector on full vehicle images.

## Files
- `PPOCR_License_Plate_Training_Submission.ipynb` — dataset preparation, grouped split, model configuration, pretrained-weight adaptation, fine-tuning and validation logs.
- `PPOCR_License_Plate_Inference_Submission.ipynb` — OCR batch evaluation, error/confidence analysis, plate detection, cropped-plate OCR, 20-image pipeline sample, and CSV generation.

## Results
| Metric | PP-OCRv4 | PP-OCRv5 |
|---|---:|---:|
| Validation exact-match (330 cropped plates) | 56.67% | 82.12% |
| Normalized edit similarity | 92.47% | 96.55% |
| 20-image detection+OCR demonstration exact-match | 45% | 90% |

The YOLO detector found plates in 20/20 demonstration images. This is not an independent end-to-end benchmark: OCR and detection splits were created independently.

## Reproduction
Use Google Colab GPU. The notebooks refer to the original Kaggle Indian vehicle dataset (`archive.zip`) and to a separately saved `PaddleOCR_Indian_Plates_Backup.zip` containing OCR training checkpoints, configs, labels, and cropped images. Upload these when prompted. The YOLO detector checkpoint `best.pt` is not bundled with these notebooks; the inference notebook includes an optional cell to train it from XML annotations. The notebooks preserve recorded outputs, but a full clean-runtime execution has not been independently reverified.

**Limitations:** Saved OCR checkpoints are `.pdparams` training weights; exported Paddle inference artifacts are not demonstrated. CUDA/NCCL compatibility may vary across Colab runtimes.
