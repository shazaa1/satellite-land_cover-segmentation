import numpy as np
import torch

CLASS_NAMES = ["Background", "Buildings", "Woodlands", "Water", "Roads"]
NUM_CLASSES = len(CLASS_NAMES)


def confusion_matrix(pred, target, n_classes=NUM_CLASSES):
    """Rows = true class, columns = predicted class."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = (target >= 0) & (target < n_classes)
    idx = n_classes * target[valid].long() + pred[valid].long()
    return torch.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_cm(cm):
    """Segmentation metrics from a confusion matrix (classes absent everywhere give NaN and are ignored in the mean)."""
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    eps = 1e-12

    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn
    iou = torch.where(iou_den > 0, tp / (iou_den + eps), torch.full_like(tp, float("nan")))
    dice = torch.where(dice_den > 0, 2 * tp / (dice_den + eps), torch.full_like(tp, float("nan")))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "mIoU": float(np.nanmean(iou.numpy())),
        "mDice": float(np.nanmean(dice.numpy())),
        "pixel_acc": float(tp.sum() / (cm.sum() + eps)),
        "per_class_IoU": iou.tolist(),
        "per_class_Dice": dice.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp=True):
    """Shared evaluation loop for all deep learning models."""
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp and device == "cuda"):
            logits = model(imgs)
        cm += confusion_matrix(logits.argmax(dim=1), masks)
    return metrics_from_cm(cm.cpu()), cm.cpu()


def print_report(m, title=""):
    print(f"=== {title} ===")
    print(f"mIoU: {m['mIoU']:.4f} | mDice: {m['mDice']:.4f} | pixel acc: {m['pixel_acc']:.4f}")
    print(f"{'class':<12}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}{m['per_class_IoU'][i]:>8.4f}{m['per_class_Dice'][i]:>8.4f}"
              f"{m['per_class_precision'][i]:>8.4f}{m['per_class_recall'][i]:>8.4f}")
