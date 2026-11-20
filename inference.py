
"""
Inference module for Satellite Land Cover Segmentation.
Supports:
- U-Net (PyTorch)
- SegFormer-B0 (TensorFlow/Keras)
- Minimal CNN (TensorFlow/Keras)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

UNET_WEIGHTS_PATH = MODELS_DIR / "unet_best.pt"
SEGFORMER_WEIGHTS_PATH = MODELS_DIR / "segformer_b0_landcover.keras"
MINIMAL_CNN_WEIGHTS_PATH = MODELS_DIR / "minimal_cnn_best.keras"

# ImageNet normalization used during Minimal CNN training
IMG_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMG_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import predict_utils

UNET_MODEL = None
UNET_DEVICE = None
SEGFORMER_MODEL = None
MINIMAL_CNN_MODEL = None


def _load_unet():
    global UNET_MODEL, UNET_DEVICE

    if UNET_MODEL is None:
        if not UNET_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"U-Net checkpoint not found: {UNET_WEIGHTS_PATH}"
            )

        UNET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading U-Net on {UNET_DEVICE}...")

        UNET_MODEL = predict_utils.load_model(
            weights_path=str(UNET_WEIGHTS_PATH),
            device=UNET_DEVICE,
        )

        print("U-Net loaded successfully.")

    return UNET_MODEL


def _load_keras_model(model_path, model_label):
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_label} checkpoint not found: {model_path}"
        )

    print(f"Loading {model_label}...")

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

    print(f"{model_label} loaded successfully.")

    return model


def _load_segformer():
    global SEGFORMER_MODEL

    if SEGFORMER_MODEL is None:
        # SegFormer was saved with KerasHub custom model classes.
        import keras_hub  # noqa: F401

        SEGFORMER_MODEL = _load_keras_model(
            SEGFORMER_WEIGHTS_PATH,
            "SegFormer-B0"
        )

    return SEGFORMER_MODEL


def _load_minimal_cnn():
    global MINIMAL_CNN_MODEL

    if MINIMAL_CNN_MODEL is None:
        MINIMAL_CNN_MODEL = _load_keras_model(
            MINIMAL_CNN_WEIGHTS_PATH,
            "Minimal CNN"
        )

    return MINIMAL_CNN_MODEL


def _keras_predict_mask(
    model,
    image_rgb,
    model_label,
    imagenet_normalization=False
):
    """
    Run a Keras segmentation model on a 512x512 RGB image.

    SegFormer-B0:
        RGB values scaled to [0, 1].

    Minimal CNN:
        ImageNet normalization:
        (image - IMG_MEAN) / IMG_STD
    """

    height, width = image_rgb.shape[:2]

    # Resize input to model input size
    resized = Image.fromarray(image_rgb).resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )

    # Convert RGB image to float32 in [0, 1]
    batch = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    # Apply ImageNet normalization only when requested
    if imagenet_normalization:
        batch = (batch - IMG_MEAN) / IMG_STD

    # Add batch dimension
    batch = batch[None, ...]

    prediction = model.predict(
        batch,
        verbose=0
    )

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if isinstance(prediction, dict):
        prediction = next(iter(prediction.values()))

    prediction = np.asarray(prediction)

    if prediction.ndim != 4 or prediction.shape[0] != 1:
        raise RuntimeError(
            f"{model_label} returned unexpected output shape "
            f"{prediction.shape}."
        )

    if prediction.shape[-1] != 5:
        raise RuntimeError(
            f"{model_label} must return 5 class channels; "
            f"received output shape {prediction.shape}."
        )

    # Convert logits/probabilities to class IDs
    mask_512 = np.argmax(
        prediction[0],
        axis=-1
    ).astype(np.uint8)

    # Resize mask back to original image dimensions
    mask = Image.fromarray(mask_512).resize(
        (width, height),
        Image.Resampling.NEAREST
    )

    return np.asarray(mask, dtype=np.uint8)


def run_inference(image: Image.Image, model_name="U-Net"):
    """
    Return:
    - class-ID mask
    - colored mask
    - overlay
    - class percentages
    - status
    """

    if image is None:
        raise ValueError("Please upload a satellite image.")

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    image = image.convert("RGB")
    image_rgb = np.asarray(
        image,
        dtype=np.uint8
    )

    if model_name == "U-Net":

        model = _load_unet()

        mask_ids = predict_utils.predict_mask(
            model=model,
            image_rgb=image_rgb,
            device=UNET_DEVICE,
        )

    elif model_name == "SegFormer-B0":

        mask_ids = _keras_predict_mask(
            model=_load_segformer(),
            image_rgb=image_rgb,
            model_label="SegFormer-B0",
            imagenet_normalization=False
        )

    elif model_name == "Minimal CNN":

        mask_ids = _keras_predict_mask(
            model=_load_minimal_cnn(),
            image_rgb=image_rgb,
            model_label="Minimal CNN",
            imagenet_normalization=True
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    mask_ids = np.asarray(
        mask_ids,
        dtype=np.uint8
    )

    colored_mask = predict_utils.colorize(mask_ids)

    overlay_image = predict_utils.overlay(
        image_rgb,
        mask_ids,
        alpha=0.45
    )

    statistics = predict_utils.class_percentages(mask_ids)

    return {
        "mask_ids": mask_ids,
        "mask": Image.fromarray(colored_mask),
        "overlay": Image.fromarray(overlay_image),
        "statistics": statistics,
        "status": (
            f"{model_name} prediction completed successfully."
        ),
    }
