
"""
Professional Three-Model Land-Cover Segmentation Comparison Demo.

Models:
- U-Net
- SegFormer-B0
- Minimal CNN

Run:
    python comparison_app.py
"""

import numpy as np
import gradio as gr
from PIL import Image

from inference import run_inference
import predict_utils


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_NAME = "LAND COVER AI"
CLASS_NAMES = [
    "Background",
    "Buildings",
    "Woodlands",
    "Water",
    "Roads",
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAMES = [
    "U-Net",
    "SegFormer-B0",
    "Minimal CNN",
]


# =========================================================
# GROUND TRUTH DECODING
# =========================================================

def decode_ground_truth(mask_image):
    """
    Supports:
    1) Grayscale/indexed masks containing class IDs 0..4.
    2) RGB masks using the project's config.json palette.
    """

    arr = np.asarray(mask_image)

    if arr.ndim == 2:
        ids = arr

    elif arr.ndim == 3 and arr.shape[2] >= 3:

        rgb = arr[:, :, :3].astype(np.uint8)

        if (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            ids = rgb[:, :, 0]

        else:
            palette = np.asarray(
                predict_utils.load_config()["palette"],
                dtype=np.uint8
            )

            ids = np.full(
                rgb.shape[:2],
                255,
                dtype=np.uint8
            )

            for class_id, color in enumerate(palette):
                ids[np.all(rgb == color, axis=-1)] = class_id

            if np.any(ids == 255):
                unique_colors = np.unique(
                    rgb.reshape(-1, 3),
                    axis=0
                )

                raise ValueError(
                    "RGB mask colours do not match config.json palette. "
                    f"Sample colours: {unique_colors[:10].tolist()}"
                )

    else:
        raise ValueError(
            "Unsupported ground-truth mask format."
        )

    ids = np.asarray(ids, dtype=np.int64)
    unique_ids = np.unique(ids)

    if np.any((ids < 0) | (ids >= NUM_CLASSES)):
        raise ValueError(
            "Ground-truth values must be class IDs 0, 1, 2, 3, 4. "
            f"Found: {unique_ids[:20].tolist()}."
        )

    return ids.astype(np.uint8)


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(y_true, y_pred):

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Mask size mismatch: ground truth {y_true.shape}, "
            f"prediction {y_pred.shape}."
        )

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    np.add.at(
        cm,
        (y_true.ravel(), y_pred.ravel()),
        1
    )

    rows = []
    ious = []
    dices = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)

        union = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn

        iou = (
            tp / union
            if union
            else float("nan")
        )

        dice = (
            (2 * tp) / dice_denominator
            if dice_denominator
            else float("nan")
        )

        if np.isfinite(iou):
            ious.append(iou)

        if np.isfinite(dice):
            dices.append(dice)

        rows.append({
            "class": class_name,
            "iou": iou,
            "dice": dice,
        })

    summary = {
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mDice": float(np.mean(dices)) if dices else 0.0,
        "Pixel Accuracy": float(np.mean(y_true == y_pred)),
    }

    return summary, rows


def _pct(value):
    return (
        f"{value * 100:.2f}%"
        if np.isfinite(value)
        else "N/A"
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

def compare_models(image, ground_truth_image):

    if image is None or ground_truth_image is None:
        raise gr.Error(
            "Please upload both the original satellite image "
            "and its matching ground-truth mask."
        )

    image = image.convert("RGB")

    y_true = decode_ground_truth(ground_truth_image)

    if y_true.shape != (image.height, image.width):
        raise gr.Error(
            "Image and ground-truth mask dimensions must match exactly. "
            f"Image: {image.width}x{image.height}; "
            f"mask: {y_true.shape[1]}x{y_true.shape[0]}."
        )

    results = {}

    # Run all three models.
    for model_name in MODEL_NAMES:

        try:
            results[model_name] = run_inference(
                image,
                model_name=model_name
            )

        except Exception as exc:
            raise gr.Error(
                f"{model_name} inference failed: {exc}"
            ) from exc

    # Print prediction pixel counts for debugging.
    for name, result in results.items():

        ids, counts = np.unique(
            result["mask_ids"],
            return_counts=True
        )

        class_counts = dict(
            zip(ids.tolist(), counts.tolist())
        )

        print(f"\n{name} prediction pixel counts:")
        print(class_counts)

    # Calculate metrics.
    summaries = {}
    per_class = {}

    for model_name in MODEL_NAMES:

        summaries[model_name], per_class[model_name] = (
            calculate_metrics(
                y_true,
                results[model_name]["mask_ids"]
            )
        )

    # Overall metrics table.
    overall_table = [
        [
            name,
            _pct(summaries[name]["mIoU"]),
            _pct(summaries[name]["mDice"]),
            _pct(summaries[name]["Pixel Accuracy"]),
        ]
        for name in MODEL_NAMES
    ]

    # Per-class metrics table.
    per_class_table = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        per_class_table.append([
            class_name,

            _pct(per_class["U-Net"][class_index]["iou"]),
            _pct(per_class["U-Net"][class_index]["dice"]),

            _pct(per_class["SegFormer-B0"][class_index]["iou"]),
            _pct(per_class["SegFormer-B0"][class_index]["dice"]),

            _pct(per_class["Minimal CNN"][class_index]["iou"]),
            _pct(per_class["Minimal CNN"][class_index]["dice"]),
        ])

    # Find the model with the highest mIoU on this image.
    best_model = max(
        MODEL_NAMES,
        key=lambda name: summaries[name]["mIoU"]
    )

    best_score = _pct(
        summaries[best_model]["mIoU"]
    )

    status = (
        f"COMPARISON COMPLETED | "
        f"Best model on this image: {best_model} "
        f"({best_score} mIoU). "
        f"These are single-image metrics, not full test-set results."
    )

    return (
        image,
        Image.fromarray(predict_utils.colorize(y_true)),
        results["U-Net"]["mask"],
        results["SegFormer-B0"]["mask"],
        results["Minimal CNN"]["mask"],
        overall_table,
        per_class_table,
        status,
    )


# =========================================================
# CUSTOM CSS
# =========================================================

custom_css = """

/* Main page */
.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    background: #0b1120 !important;
    color: #f1f5f9 !important;
    padding: 24px !important;
}

/* General blocks */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box {
    background: #151f32 !important;
    border: 1px solid #29384f !important;
    border-radius: 14px !important;
}

/* Text */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label,
.gradio-container span {
    color: #f1f5f9 !important;
}

/* Header */
#main-header {
    background: linear-gradient(
        120deg,
        #14233d,
        #172c4d,
        #173b5d
    ) !important;

    border: 1px solid #294a70 !important;
    border-radius: 20px !important;
    padding: 28px !important;
    margin-bottom: 22px !important;
}

/* Header title */
#main-header h1 {
    font-size: 34px !important;
    font-weight: 850 !important;
    letter-spacing: 1px;
    margin-bottom: 8px !important;
}

/* Header subtitle */
#main-header p {
    color: #b9cce3 !important;
    font-size: 15px !important;
}

/* Section headings */
.section-title {
    font-size: 22px !important;
    font-weight: 800 !important;
    margin-top: 20px !important;
    margin-bottom: 12px !important;
}

/* Model cards */
.model-card {
    background: #151f32 !important;
    border: 1px solid #30425d !important;
    border-radius: 15px !important;
    padding: 14px !important;
}

/* Model titles */
.model-card h3 {
    font-size: 17px !important;
    font-weight: 750 !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(
        90deg,
        #2563eb,
        #4f46e5
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 750 !important;
    min-height: 48px !important;
    transition: 0.2s ease !important;
}

button.primary:hover {
    filter: brightness(1.15);
    transform: translateY(-1px);
}

/* Secondary buttons */
button.secondary {
    border-radius: 10px !important;
}

/* Images */
.image-container,
.gr-image {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Tables */
table {
    border-radius: 10px !important;
    overflow: hidden !important;
}

thead th {
    background: #1d2c43 !important;
    color: #93c5fd !important;
    font-weight: 800 !important;
    border-bottom: 1px solid #3b4e6b !important;
}

tbody td {
    background: #111b2c !important;
    color: #e2e8f0 !important;
    border-bottom: 1px solid #27364c !important;
}

/* Status */
#status-box textarea {
    background: #102b3b !important;
    color: #86efac !important;
    border: 1px solid #28634d !important;
    font-weight: 650 !important;
}

/* Upload components */
.upload-box {
    border: 1px dashed #47658b !important;
    border-radius: 14px !important;
}

/* Footer */
footer {
    display: none !important;
}

/* Responsive */
@media (max-width: 900px) {
    .gradio-container {
        padding: 12px !important;
    }

    #main-header {
        padding: 18px !important;
    }

    #main-header h1 {
        font-size: 25px !important;
    }

    .section-title {
        font-size: 19px !important;
    }
}

"""


# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="Land Cover AI | Three-Model Comparison",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
    css=custom_css,
) as demo:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    gr.HTML(
        """
        <div id="main-header">
            <div style="
                display:flex;
                align-items:center;
                gap:12px;
                margin-bottom:14px;
            ">
                <div style="
                    background:#2563eb;
                    width:48px;
                    height:48px;
                    border-radius:13px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:27px;
                ">
                    🛰️
                </div>

                <div>
                    <div style="
                        color:#93c5fd;
                        font-size:12px;
                        font-weight:800;
                        letter-spacing:2px;
                    ">
                        AI / COMPUTER VISION PROJECT
                    </div>

                    <div style="
                        color:#ffffff;
                        font-size:15px;
                        font-weight:700;
                    ">
                        LAND COVER ANALYSIS PLATFORM
                    </div>
                </div>
            </div>

            <h1>Satellite Image Segmentation</h1>

            <p>
                Intelligent land-cover classification using deep learning.
                Compare three segmentation architectures on the same
                satellite image and ground-truth mask.
            </p>

            <div style="
                display:flex;
                flex-wrap:wrap;
                gap:9px;
                margin-top:18px;
            ">
                <span style="
                    background:#1e3a5f;
                    color:#bfdbfe;
                    padding:7px 12px;
                    border-radius:20px;
                    font-size:12px;
                    font-weight:700;
                ">
                    ● U-Net
                </span>

                <span style="
                    background:#164e63;
                    color:#a5f3fc;
                    padding:7px 12px;
                    border-radius:20px;
                    font-size:12px;
                    font-weight:700;
                ">
                    ● SegFormer-B0
                </span>

                <span style="
                    background:#37306b;
                    color:#ddd6fe;
                    padding:7px 12px;
                    border-radius:20px;
                    font-size:12px;
                    font-weight:700;
                ">
                    ● Minimal CNN
                </span>
            </div>
        </div>
        """
    )

    # -----------------------------------------------------
    # INPUT SECTION
    # -----------------------------------------------------

    gr.Markdown(
        "## 01 — Input Data",
        elem_classes=["section-title"]
    )

    gr.Markdown(
        "Upload the satellite image and its corresponding ground-truth mask."
    )

    with gr.Row(equal_height=True):

        with gr.Column(
            scale=1,
            elem_classes=["upload-box"]
        ):

            input_image = gr.Image(
                type="pil",
                label="Original Satellite Image",
                height=300,
            )

        with gr.Column(
            scale=1,
            elem_classes=["upload-box"]
        ):

            ground_truth = gr.Image(
                type="pil",
                label="Ground-Truth Mask",
                height=300,
            )

    compare_button = gr.Button(
        "▶  RUN THREE-MODEL COMPARISON",
        variant="primary",
        size="lg",
    )

    # -----------------------------------------------------
    # VISUAL RESULTS
    # -----------------------------------------------------

    gr.Markdown(
        "## 02 — Segmentation Results",
        elem_classes=["section-title"]
    )

    gr.Markdown(
        "Compare the original image, ground truth, and predictions."
    )

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            original_output = gr.Image(
                label="Original Image",
                height=300,
                interactive=False,
            )

        with gr.Column(scale=1):
            gt_output = gr.Image(
                label="Ground Truth",
                height=300,
                interactive=False,
            )

    gr.Markdown(
        "### Model Predictions"
    )

    with gr.Row(equal_height=True):

        with gr.Column(
            scale=1,
            elem_classes=["model-card"]
        ):

            gr.Markdown(
                """
                ### U-Net
                <span style="color:#93c5fd">
                PyTorch segmentation model
                </span>
                """
            )

            unet_output = gr.Image(
                label="U-Net Prediction",
                height=320,
                interactive=False,
            )

        with gr.Column(
            scale=1,
            elem_classes=["model-card"]
        ):

            gr.Markdown(
                """
                ### SegFormer-B0
                <span style="color:#67e8f9">
                Transformer-based segmentation
                </span>
                """
            )

            segformer_output = gr.Image(
                label="SegFormer-B0 Prediction",
                height=320,
                interactive=False,
            )

        with gr.Column(
            scale=1,
            elem_classes=["model-card"]
        ):

            gr.Markdown(
                """
                ### Minimal CNN
                <span style="color:#c4b5fd">
                Lightweight CNN baseline
                </span>
                """
            )

            minimal_cnn_output = gr.Image(
                label="Minimal CNN Prediction",
                height=320,
                interactive=False,
            )

    # -----------------------------------------------------
    # OVERALL METRICS
    # -----------------------------------------------------

    gr.Markdown(
        "## 03 — Overall Performance",
        elem_classes=["section-title"]
    )

    gr.Markdown(
        "Quantitative comparison of all three models."
    )

    overall_table = gr.Dataframe(
        headers=[
            "Model",
            "mIoU",
            "mDice",
            "Pixel Accuracy",
        ],
        datatype=[
            "str",
            "str",
            "str",
            "str",
        ],
        interactive=False,
        label="Overall Model Comparison",
        elem_classes=["metrics-table"],
    )

    # -----------------------------------------------------
    # PER CLASS METRICS
    # -----------------------------------------------------

    gr.Markdown(
        "## 04 — Per-Class Performance",
        elem_classes=["section-title"]
    )

    gr.Markdown(
        "Detailed IoU and Dice comparison for each land-cover class."
    )

    per_class_table = gr.Dataframe(
        headers=[
            "Class",
            "U-Net IoU",
            "U-Net Dice",
            "SegFormer IoU",
            "SegFormer Dice",
            "Minimal CNN IoU",
            "Minimal CNN Dice",
        ],
        datatype=["str"] * 7,
        interactive=False,
        label="Per-Class Model Comparison",
        elem_classes=["metrics-table"],
    )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    gr.Markdown(
        "## 05 — Analysis Status",
        elem_classes=["section-title"]
    )

    status = gr.Textbox(
        label="Comparison Summary",
        interactive=False,
        lines=3,
        elem_id="status-box",
        placeholder="Upload both images and run the comparison.",
    )

    # -----------------------------------------------------
    # EVENT
    # -----------------------------------------------------

    compare_button.click(
        fn=compare_models,
        inputs=[
            input_image,
            ground_truth,
        ],
        outputs=[
            original_output,
            gt_output,
            unet_output,
            segformer_output,
            minimal_cnn_output,
            overall_table,
            per_class_table,
            status,
        ],
        show_progress="full",
    )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    gr.HTML(
        """
        <div style="
            margin-top:30px;
            padding:20px;
            border-top:1px solid #29384f;
            text-align:center;
            color:#94a3b8;
            font-size:12px;
        ">
            <div style="
                color:#cbd5e1;
                font-weight:750;
                letter-spacing:1px;
                margin-bottom:7px;
            ">
                LAND COVER AI
            </div>

            Satellite Image Segmentation & Three-Model Comparison

            <div style="margin-top:7px;">
                U-Net · SegFormer-B0 · Minimal CNN
            </div>
        </div>
        """
    )

    gr.Markdown(
        """
        **Evaluation note:** Metrics displayed here are calculated only
        for the uploaded image. Final model ranking should be based on
        evaluation over the complete held-out test dataset.
        """
    )


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":

    demo.launch()
