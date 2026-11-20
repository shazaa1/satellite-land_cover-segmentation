
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import sys

MODEL_PATH = Path("models/minimal_cnn_best.keras")

CLASS_NAMES = [
    "Background",
    "Buildings",
    "Woodlands",
    "Water",
    "Roads"
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def calculate_metrics(y_true, y_pred):
    num_classes = 5
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    np.add.at(cm, (y_true.ravel(), y_pred.ravel()), 1)

    ious = []
    dices = []

    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        union = tp + fp + fn
        denominator = 2 * tp + fp + fn

        iou = tp / union if union else np.nan
        dice = 2 * tp / denominator if denominator else np.nan

        if np.isfinite(iou):
            ious.append(iou)

        if np.isfinite(dice):
            dices.append(dice)

        print(
            f"{name:12s} | "
            f"IoU: {iou * 100:.2f}% | "
            f"Dice: {dice * 100:.2f}%"
            if np.isfinite(iou)
            else f"{name:12s} | IoU: N/A | Dice: N/A"
        )

    print(f"mIoU: {(np.mean(ious) * 100):.2f}%")
    print(f"mDice: {(np.mean(dices) * 100):.2f}%")
    print(f"Pixel Accuracy: {np.mean(y_true == y_pred) * 100:.2f}%")


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print('python diagnose_minimal_cnn.py "IMAGE_PATH" "MASK_PATH"')
        return

    image_path = sys.argv[1]
    mask_path = sys.argv[2]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image, dtype=np.uint8)

    mask = np.asarray(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint8)

    if image_rgb.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions do not match.")

    if np.any(mask >= 5):
        raise ValueError(
            f"Mask must contain class IDs 0-4. Found: {np.unique(mask)}"
        )

    print("Loading Minimal CNN...")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded:", model.name)
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)

    rgb = image_rgb.astype(np.float32) / 255.0

    preprocessing = {
        "RGB /255": rgb,
        "ImageNet Normalization": (rgb - MEAN) / STD
    }

    print("\nGround Truth distribution:")
    ids, counts = np.unique(mask, return_counts=True)
    print({
        CLASS_NAMES[int(i)]: int(c)
        for i, c in zip(ids, counts)
    })

    for name, processed_image in preprocessing.items():
        print("\n" + "=" * 55)
        print("PREPROCESSING:", name)

        prediction = model.predict(
            processed_image[None, ...],
            verbose=0
        )

        prediction_mask = np.argmax(
            prediction[0],
            axis=-1
        ).astype(np.uint8)

        ids, counts = np.unique(
            prediction_mask,
            return_counts=True
        )

        print("\nPrediction distribution:")
        print({
            CLASS_NAMES[int(i)]: int(c)
            for i, c in zip(ids, counts)
        })

        print("\nMetrics:")
        calculate_metrics(mask, prediction_mask)

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
