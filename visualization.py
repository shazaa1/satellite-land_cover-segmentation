
"""
Visualization Module
Satellite Land Cover Segmentation

Responsible for:
1. Generating colored segmentation masks.
2. Creating overlay images.
3. Calculating land-cover class percentages.
"""

import numpy as np
from PIL import Image


# Six land-cover classes
CLASS_NAMES = {
    0: "Forest",
    1: "Water",
    2: "Agriculture",
    3: "Buildings",
    4: "Bare Soil",
    5: "Roads"
}


# RGB colors for each class
CLASS_COLORS = {
    0: (34, 197, 94),     # Forest - Green
    1: (59, 130, 246),    # Water - Blue
    2: (250, 204, 21),    # Agriculture - Yellow
    3: (239, 68, 68),     # Buildings - Red
    4: (139, 90, 80),     # Bare Soil - Brown
    5: (216, 180, 239)    # Roads - Light Purple
}


def create_colored_mask(mask):
    """
    Convert a class-ID mask into a colored RGB image.

    Args:
        mask: 2D numpy array containing class IDs.

    Returns:
        PIL RGB image.
    """

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    colored_mask = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8
    )

    for class_id, color in CLASS_COLORS.items():
        colored_mask[mask == class_id] = color

    return Image.fromarray(colored_mask, mode="RGB")


def create_overlay(original_image, mask, alpha=0.45):
    """
    Blend the colored segmentation mask with the original image.

    Args:
        original_image: Original satellite image.
        mask: 2D class-ID segmentation mask.
        alpha: Mask transparency.

    Returns:
        PIL RGB overlay image.
    """

    if original_image is None:
        raise ValueError("Original image is required.")

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1.")

    original_image = original_image.convert("RGB")

    colored_mask = create_colored_mask(mask)

    colored_mask = colored_mask.resize(
        original_image.size,
        Image.Resampling.NEAREST
    )

    return Image.blend(
        original_image,
        colored_mask,
        alpha
    )


def calculate_statistics(mask):
    """
    Calculate the percentage of pixels belonging to each class.

    Args:
        mask: 2D numpy array containing class IDs.

    Returns:
        Dictionary containing class percentages.
    """

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    total_pixels = mask.size

    if total_pixels == 0:
        raise ValueError("Mask is empty.")

    statistics = {}

    for class_id, class_name in CLASS_NAMES.items():

        class_pixels = np.sum(mask == class_id)

        percentage = (
            class_pixels / total_pixels
        ) * 100

        statistics[class_name] = round(
            float(percentage), 2
        )

    return statistics
