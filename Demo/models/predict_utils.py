"""Helpers for the demo: load the U-Net, predict a mask for any aerial image, colour it, count the classes."""
import json
import os

import cv2
import numpy as np
import torch

from unet_model import UNet

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, "config.json")) as f:
        return json.load(f)


def load_model(weights_path=None, device="cpu"):
    cfg = load_config()
    model = UNet(n_classes=len(cfg["class_names"]))
    model.load_state_dict(torch.load(weights_path or os.path.join(HERE, "unet_best.pt"), map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_mask(model, image_rgb, device="cpu"):
    """image_rgb: uint8 array (H, W, 3). Returns a uint8 mask (H, W) with the class id of every pixel.
    The image is cut into 512x512 tiles; every tile is resized to the network size, exactly like in training."""
    cfg = load_config()
    img_size, tile = cfg["img_size"], cfg["patch_size"]
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)

    h, w = image_rgb.shape[:2]
    padded = cv2.copyMakeBorder(image_rgb, 0, (-h) % tile, 0, (-w) % tile, cv2.BORDER_REPLICATE)
    out = np.zeros(padded.shape[:2], dtype=np.uint8)
    for y in range(0, padded.shape[0], tile):
        for x in range(0, padded.shape[1], tile):
            patch = cv2.resize(padded[y:y + tile, x:x + tile], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            tensor = ((patch.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
            logits = model(torch.from_numpy(tensor).unsqueeze(0).to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            out[y:y + tile, x:x + tile] = cv2.resize(pred, (tile, tile), interpolation=cv2.INTER_NEAREST)
    return out[:h, :w]


def colorize(mask):
    """Class-id mask -> RGB image."""
    return np.array(load_config()["palette"], dtype=np.uint8)[mask]


def overlay(image_rgb, mask, alpha=0.45):
    """Mask drawn on top of the image."""
    return (image_rgb * (1 - alpha) + colorize(mask) * alpha).astype(np.uint8)


def class_percentages(mask):
    """Share (%) of the pixels of every class."""
    names = load_config()["class_names"]
    counts = np.bincount(mask.reshape(-1), minlength=len(names))
    return {n: float(c) / mask.size * 100 for n, c in zip(names, counts)}
