"""
Satellite Land Cover Segmentation
Gradio application supporting PyTorch U-Net and TensorFlow SegFormer-B0.
"""

import tempfile
from pathlib import Path

import gradio as gr
from inference import run_inference

PROJECT_NAME = "Satellite Land Cover Segmentation"
DASHES = ["—"] * 5


# ================= THEME SYSTEM =================

def change_theme(mode):
    if mode == "Dark":
        return """
        <style>
        body, .gradio-container {
            background: #0b1120 !important;
            color: #f1f5f9 !important;
            color-scheme: dark !important;
        }
        .gradio-container .block,
        .gradio-container .form {
            background: #172235 !important;
            border-color: #29384f !important;
        }
        .gradio-container .prose,
        .gradio-container .prose *,
        .gradio-container label,
        .gradio-container label span {
            color: #f1f5f9 !important;
        }
        .gradio-container input,
        .gradio-container select {
            background: #101a2b !important;
            color: #f1f5f9 !important;
        }
        .gradio-container button.primary {
            background: #635bff !important;
            color: white !important;
        }
        .gradio-container .nav-link {
            color: #cbd5e1 !important;
        }
        </style>
        """

    return """
    <style>
    body, .gradio-container {
        background: #f1f5f9 !important;
        color: #172033 !important;
        color-scheme: light !important;
    }
    .gradio-container .block,
    .gradio-container .form {
        background: #ffffff !important;
        border-color: #dce4ef !important;
        color: #172033 !important;
    }
    .gradio-container .prose,
    .gradio-container .prose *,
    .gradio-container label,
    .gradio-container label span {
        color: #172033 !important;
    }
    .gradio-container input,
    .gradio-container select {
        background: #ffffff !important;
        color: #172033 !important;
    }
    .gradio-container button.primary {
        background: #635bff !important;
        color: white !important;
    }
    .gradio-container .nav-link {
        color: #334155 !important;
    }
    </style>
    """


# ================= IMAGE UPLOAD =================

def upload_image(image):
    if image is None:
        return None, "📤 No image uploaded yet. Please upload a satellite image."
    return image, "✅ Image uploaded successfully. Ready for analysis."


# ================= INFERENCE =================

def analyze_image(image, model_choice):
    if image is None:
        return (
            None, None, None,
            *DASHES,
            None, None,
            "⚠️ Please upload a satellite image first."
        )

    mask_file = None
    overlay_file = None

    try:
        result = run_inference(image, model_name=model_choice)

        colored_mask = result.get("mask")
        overlay = result.get("overlay")
        stats = result.get("statistics") or {}

        if colored_mask is None or overlay is None:
            raise RuntimeError(result.get("status", "The model did not return prediction images."))

        class_values = [
            stats.get("Background", 0.0),
            stats.get("Buildings", 0.0),
            stats.get("Woodlands", 0.0),
            stats.get("Water", 0.0),
            stats.get("Roads", 0.0),
        ]
        formatted_values = [f"{value:.2f}%" for value in class_values]

        with tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False) as temp_mask:
            mask_file = temp_mask.name
        colored_mask.save(mask_file)

        with tempfile.NamedTemporaryFile(suffix="_overlay.png", delete=False) as temp_overlay:
            overlay_file = temp_overlay.name
        overlay.save(overlay_file)

        return (
            image,
            colored_mask,
            overlay,
            *formatted_values,
            mask_file,
            overlay_file,
            f"✅ {model_choice} segmentation completed successfully."
        )

    except Exception as error:
        for path in (mask_file, overlay_file):
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except OSError:
                    pass

        return (
            image, None, None,
            *DASHES,
            None, None,
            f"❌ Analysis failed: {error}"
        )


# ================= CUSTOM CSS =================

custom_css = """
#theme-style {
    display: none !important;
}
html, body {
    width: 100% !important;
    margin: 0 !important;
}
.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 20px 3% !important;
    box-sizing: border-box !important;
}
.header-title {
    font-size: 30px !important;
    font-weight: 800 !important;
}
.nav-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    border-radius: 12px;
    margin: 8px 0 20px 0;
    background: rgba(128, 128, 128, 0.10);
}
.nav-link {
    text-decoration: none !important;
    font-weight: 700;
    font-size: 14px;
    padding: 8px 12px;
    border-radius: 7px;
    transition: 0.2s;
}
.nav-link:hover {
    background: rgba(99, 91, 255, 0.15);
    color: #8179ff !important;
}
.nav-separator {
    opacity: 0.4;
}
button.primary {
    border-radius: 9px !important;
    font-weight: 700 !important;
}
footer {
    display: none !important;
}
@media (max-width: 768px) {
    .gradio-container {
        padding: 12px !important;
    }
    .header-title {
        font-size: 23px !important;
    }
    .nav-bar {
        gap: 5px;
        padding: 8px;
    }
    .nav-link {
        font-size: 12px;
        padding: 6px;
    }
}
"""


# ================= MAIN APPLICATION =================

with gr.Blocks(
    title=PROJECT_NAME,
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    theme_styles = gr.HTML(value=change_theme("Dark"), elem_id="theme-style")

    # HEADER
    with gr.Row(equal_height=True, elem_id="overview"):
        with gr.Column(scale=4):
            gr.Markdown("""
            # 🛰️ SATELLITE AI

            **LAND COVER SEGMENTATION PLATFORM**

            AI-powered satellite image analysis and land-cover identification.
            """)
        with gr.Column(scale=1):
            theme_choice = gr.Radio(
                choices=["Dark", "Light"],
                value="Dark",
                label="🎨 Appearance"
            )

    theme_choice.change(
        fn=change_theme,
        inputs=theme_choice,
        outputs=theme_styles
    )

    gr.Markdown("---")

    # NAVIGATION
    gr.HTML("""
    <nav class="nav-bar">
        <a class="nav-link" href="#overview">⌂ Overview</a>
        <span class="nav-separator">|</span>
        <a class="nav-link" href="#image-analysis">📤 Image Analysis</a>
        <span class="nav-separator">|</span>
        <a class="nav-link" href="#results">🧠 Results</a>
        <span class="nav-separator">|</span>
        <a class="nav-link" href="#about">ℹ About</a>
    </nav>
    """)

    # IMAGE ANALYSIS
    gr.Markdown("## 📤 Image Analysis", elem_id="image-analysis")
    gr.Markdown("Upload a satellite image and select a model for land-cover segmentation.")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="UPLOAD SATELLITE IMAGE",
                height=400
            )
            model_choice = gr.Radio(
                choices=["U-Net", "SegFormer-B0"],
                value="U-Net",
                label="🧠 SELECT SEGMENTATION MODEL"
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Satellite Image",
                variant="primary",
                size="lg"
            )
            status = gr.Textbox(
                label="SYSTEM STATUS",
                value="📤 No image uploaded yet. Please upload a satellite image.",
                interactive=False,
                lines=2
            )

        with gr.Column(scale=1):
            original_image = gr.Image(
                label="ORIGINAL SATELLITE IMAGE",
                height=400,
                interactive=False
            )

    input_image.change(
        fn=upload_image,
        inputs=input_image,
        outputs=[original_image, status]
    )

    # SEGMENTATION RESULTS
    gr.Markdown("---")
    gr.Markdown("## 🧠 Segmentation Results", elem_id="results")
    gr.Markdown("Compare the original image with the predicted segmentation outputs.")

    with gr.Row():
        predicted_mask = gr.Image(
            label="PREDICTED SEGMENTATION MASK",
            height=350,
            interactive=False
        )
        overlay_image = gr.Image(
            label="SEGMENTATION OVERLAY",
            height=350,
            interactive=False
        )

    # STATISTICS
    gr.Markdown("## 📊 Land Cover Statistics")
    gr.Markdown("Pixel distribution across the five classes predicted by the selected model.")

    with gr.Row():
        background = gr.Textbox(value="—", label="⚪ Background (%)", interactive=False)
        buildings = gr.Textbox(value="—", label="🔴 Buildings (%)", interactive=False)
        woodlands = gr.Textbox(value="—", label="🟢 Woodlands (%)", interactive=False)

    with gr.Row():
        water = gr.Textbox(value="—", label="🔵 Water (%)", interactive=False)
        roads = gr.Textbox(value="—", label="🟡 Roads (%)", interactive=False)

    # CLASSIFICATION LEGEND
    gr.Markdown("---")
    gr.Markdown("## 🌍 Land Cover Classification")

    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ⚪ **Background**

            🔴 **Buildings**

            🟢 **Woodlands**
            """)
        with gr.Column():
            gr.Markdown("""
            🔵 **Water**

            🟡 **Roads**
            """)

    # EXPORT
    gr.Markdown("---")
    gr.Markdown("## 📥 Export Results")

    with gr.Row():
        mask_download = gr.File(label="SEGMENTATION MASK", interactive=False)
        overlay_download = gr.File(label="OVERLAY IMAGE", interactive=False)

    # ABOUT
    gr.Markdown("---")
    gr.Markdown("## 🛰️ About Satellite AI", elem_id="about")
    gr.Markdown("""
    A Deep Learning application designed to perform pixel-level segmentation of satellite imagery.

    **Available models:** U-Net (PyTorch) and SegFormer-B0 (TensorFlow/Keras), trained on five land-cover classes.

    **Deep Learning Final Project | Fall 2026**
    """)

    # ANALYZE BUTTON CONNECTION
    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image, model_choice],
        outputs=[
            original_image,
            predicted_mask,
            overlay_image,
            background,
            buildings,
            woodlands,
            water,
            roads,
            mask_download,
            overlay_download,
            status
        ]
    )


# ================= LAUNCH =================

if __name__ == "__main__":
    demo.launch()
