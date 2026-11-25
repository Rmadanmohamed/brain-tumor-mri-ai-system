"""Audit the existing MRI classifier without changing model weights.
Run from the project directory: python audit_classifier.py
Requires the project's existing Keras/TensorFlow environment and scikit-learn.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mri_model import CLASS_NAMES, MODEL_PATH, load_classifier, preprocess_image

FOLDERS = ("glioma", "meningioma", "notumor", "pituitary")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--testing", type=Path, default=Path(__file__).resolve().parent / "dataset" / "Testing")
    parser.add_argument("--image", type=Path, help="Optional exact image to test before dataset audit")
    parser.add_argument("--single-only", action="store_true", help="Only evaluate --image")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("classifier_audit"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.single_only and not args.image:
        parser.error("--single-only requires --image")
    model = load_classifier(args.model)
    def predict(paths):
        arrays = []
        for path in paths:
            with Image.open(path) as img:
                arrays.append(preprocess_image(img)[0])
        result = np.asarray(model.predict(np.stack(arrays), verbose=0))
        if (result.shape != (len(paths), len(CLASS_NAMES))
            or not np.isfinite(result).all()
            or np.any(result < 0) or np.any(result > 1)
            or not np.allclose(result.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Invalid classifier output")
        return result
    if args.image:
        p = predict([args.image])[0]
        print(f"\nImage: {args.image}")
        for label, score in zip(CLASS_NAMES, p):
            print(f"{label}: {score:.2%}")
        print("Top class:", CLASS_NAMES[int(np.argmax(p))])
    if args.single_only:
        return
    paths, truth = [], []
    for i, folder in enumerate(FOLDERS):
        directory = args.testing / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing expected class folder: {directory}")
        images = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f"No images in {directory}")
        paths.extend(images)
        truth.extend([i] * len(images))
    probabilities = []
    for start in range(0, len(paths), args.batch_size):
        probabilities.extend(predict(paths[start:start+args.batch_size]))
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(truth, predicted, labels=labels, target_names=CLASS_NAMES, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    summary = (
        f"Images: {len(paths)}\nAccuracy vs dataset labels: {accuracy_score(truth, predicted):.4%}\n\n"
        f"{report}\nConfusion matrix (rows=true, columns=predicted):\n{matrix}\n\n"
        "Caution: the original training notebook uses the full Testing folder for validation. "
        "These scores are diagnostic, not an independent estimate of generalization or clinical accuracy.\n"
    )
    print(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.txt").write_text(summary, encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_class", "predicted_class", "correct", *CLASS_NAMES])
        for path, y, pred, scores in zip(paths, truth, predicted, probabilities):
            writer.writerow([str(path), CLASS_NAMES[y], CLASS_NAMES[pred], int(y == pred), *map(float, scores)])
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true / predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row])
    print("Reports saved to:", args.output.resolve())

if __name__ == "__main__":
    main()
