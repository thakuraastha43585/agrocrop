"""AgroVision AI — Smart Crop Disease Detection Streamlit Web Application."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import format_class_label
from src.predict import (
    InvalidImageError,
    load_model_and_classes,
    predict_image,
    validate_and_load_image,
)

st.set_page_config(
    page_title="AgroVision AI | Smart Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_styles() -> None:
    """Apply a clean green, white, and light-neutral agricultural AI theme."""
    st.markdown(
        """
        <style>
        .hero-banner {
            background: linear-gradient(135deg, #e8f5e9 0%, #f6faf6 100%);
            border: 1px solid #c8e6c9;
            border-left: 6px solid #2e7d32;
            border-radius: 10px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }
        .hero-banner h1 {
            color: #1b5e20;
            margin: 0 0 0.4rem 0;
            font-size: 2rem;
        }
        .hero-banner p {
            color: #2f4534;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .info-card {
            background-color: #f7faf7;
            border: 1px solid #dce8dd;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            height: 100%;
        }
        .info-card h4 {
            color: #2e7d32;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .prediction-box {
            background-color: #f1f8e9;
            border: 1px solid #aed581;
            border-left: 6px solid #2e7d32;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .prediction-healthy {
            background-color: #e8f5e9;
            border-color: #81c784;
            border-left-color: #2e7d32;
        }
        .prediction-disease {
            background-color: #fff8e1;
            border-color: #ffe082;
            border-left-color: #ef6c00;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained MobileNetV2 model...")
def get_cached_model():
    """Load and cache the trained Keras model and class labels across reruns."""
    return load_model_and_classes(config.MODEL_PATH, config.CLASS_NAMES_PATH)


@st.cache_data
def load_json_report(path_str: str) -> dict | None:
    """Read a JSON report artifact if present on disk."""
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_disclaimer() -> None:
    """Display the experimental-use disclaimer consistently."""
    st.warning(
        "**Experimental Disclaimer:** AgroVision AI is an academic deep learning project "
        "trained on the PlantVillage dataset. Predictions are experimental and should not "
        "be used as a substitute for professional agronomic diagnosis or crop treatment advice."
    )


def render_home_section(eval_results: dict | None, model_config: dict | None) -> None:
    """Render the Home overview, project objectives, and UN SDG alignment."""
    st.markdown(
        """
        <div class="hero-banner">
            <h1>AgroVision AI — Smart Crop Disease Detection</h1>
            <p>
                An image-based plant pathology screening system powered by <strong>TensorFlow/Keras</strong>
                and <strong>MobileNetV2 transfer learning</strong> to classify tomato leaf diseases
                quickly and transparently.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if eval_results and model_config:
        splits = model_config.get("split_counts", {})
        total_imgs = sum(splits.values()) if splits else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target Crop", "Tomato Leaf")
        c2.metric("Disease Classes", f"{model_config.get('num_classes', 5)}")
        c3.metric("Verified Dataset", f"{total_imgs:,} images")
        c4.metric(
            "Held-Out Test Accuracy",
            f"{eval_results['test_accuracy'] * 100:.2f}%",
            help=f"Measured on {eval_results['test_samples']:,} unseen test images",
        )

    st.subheader("Project Objectives")
    st.markdown(
        """
        - **Accurate Leaf Disease Classification:** Identify five tomato leaf conditions
          (`Tomato Bacterial Spot`, `Tomato Early Blight`, `Tomato Late Blight`,
          `Tomato Leaf Mold`, and `Tomato Healthy`) from leaf photographs.
        - **Transfer Learning Efficiency:** Leverage ImageNet-pretrained **MobileNetV2**
          features so training and inference remain lightweight and fast on standard CPU hardware.
        - **Leakage-Free & Class-Balanced Pipeline:** Use stratified `70/15/15` dataset
          splitting and Scikit-learn balanced class weights to prevent bias toward majority classes.
        - **Transparent Evaluation:** Report real, reproducible metrics on unseen test data
          alongside a clean web interface for live image testing.
        """
    )

    st.subheader("Alignment with UN Sustainable Development Goals (SDGs)")
    col_sdg2, col_sdg13 = st.columns(2)
    with col_sdg2:
        st.markdown(
            """
            <div class="info-card">
                <h4>SDG 2 — Zero Hunger</h4>
                <p>
                    Plant diseases can destroy significant portions of a harvest if detected too late.
                    Accessible early screening helps farmers identify foliar infections sooner,
                    protect crop yields, and strengthen local food security.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_sdg13:
        st.markdown(
            """
            <div class="info-card">
                <h4>SDG 13 — Climate Action</h4>
                <p>
                    Accurate disease identification supports targeted crop care rather than
                    blanket chemical spraying, reducing unnecessary pesticide/fungicide runoff
                    and lowering agriculture's environmental footprint under changing climates.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    render_disclaimer()


def render_detection_section() -> None:
    """Render image upload, real MobileNetV2 inference, and explanation."""
    st.header("Tomato Leaf Disease Detection")
    st.write(
        "Upload a clear photograph of a tomato leaf (`JPG`, `JPEG`, or `PNG`) and click "
        "**Detect Disease** to run live inference with the trained MobileNetV2 model."
    )

    uploaded_file = st.file_uploader(
        "Upload Tomato Leaf Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG.",
    )

    if uploaded_file is None:
        st.info("Upload a tomato leaf image above to begin analysis.")
        render_disclaimer()
        return

    try:
        pil_image = validate_and_load_image(uploaded_file)
    except InvalidImageError as exc:
        st.error(f"Invalid image file: {exc}")
        render_disclaimer()
        return

    col_img, col_res = st.columns([1, 1.2], gap="large")
    with col_img:
        st.subheader("Uploaded Leaf Preview")
        st.image(
            pil_image,
            caption=f"{uploaded_file.name} ({pil_image.size[0]}×{pil_image.size[1]} px)",
            width="stretch",
        )

    with col_res:
        st.subheader("Inference Result")
        detect_clicked = st.button("Detect Disease", type="primary", width="stretch")

        if detect_clicked:
            try:
                model, class_names = get_cached_model()
                with st.spinner("Running MobileNetV2 inference..."):
                    result = predict_image(pil_image, model=model, class_names=class_names)
                st.session_state["last_prediction"] = result
                st.session_state["last_filename"] = uploaded_file.name
            except FileNotFoundError as exc:
                st.error(f"Model artifact missing: {exc}")
                return
            except InvalidImageError as exc:
                st.error(f"Invalid image input: {exc}")
                return
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        saved_pred = st.session_state.get("last_prediction")
        if saved_pred and st.session_state.get("last_filename") == uploaded_file.name:
            is_healthy = saved_pred["predicted_class"] == "Tomato_healthy"
            box_class = "prediction-healthy" if is_healthy else "prediction-disease"
            st.markdown(
                f"""
                <div class="prediction-box {box_class}">
                    <div style="font-size:0.85rem; color:#4b5e4e; font-weight:600;">
                        PREDICTED CONDITION
                    </div>
                    <div style="font-size:1.55rem; font-weight:700; color:#1b2a1e; margin:0.2rem 0;">
                        {saved_pred['display_name']}
                    </div>
                    <div style="font-size:1.05rem; color:#2e7d32; font-weight:600;">
                        Confidence: {saved_pred['confidence_pct']:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            info = saved_pred["disease_info"]
            st.markdown(f"**Likely Cause:** {info['cause']}")
            st.markdown(f"**Overview:** {info['summary']}")
            st.markdown(f"**General Note:** {info['guidance']}")

            st.markdown("**Class Probability Breakdown**")
            for item in saved_pred["probabilities"]:
                st.progress(
                    min(max(item["probability"], 0.0), 1.0),
                    text=f"{item['display_name']} — {item['percentage']:.2f}%",
                )
        else:
            st.info("Click **Detect Disease** to analyze the uploaded leaf image.")

    st.markdown("")
    render_disclaimer()


def render_performance_section(eval_results: dict | None) -> None:
    """Render real test-set metrics and training/evaluation plots."""
    st.header("Model Performance (Held-Out Test Set)")

    if eval_results is None:
        st.error(
            f"Evaluation results not found at `{config.EVALUATION_RESULTS_PATH}`. "
            "Run `python -m src.evaluate` first."
        )
        return

    macro = eval_results["macro_avg"]
    weighted = eval_results["weighted_avg"]

    st.caption(
        f"Evaluated strictly on `{eval_results['test_samples']:,}` unseen test images "
        f"using `{eval_results['model_path']}`."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{eval_results['test_accuracy'] * 100:.2f}%")
    m2.metric("Test Loss", f"{eval_results['test_loss']:.4f}")
    m3.metric("Macro F1-Score", f"{macro['f1_score'] * 100:.2f}%")
    m4.metric("Weighted F1-Score", f"{weighted['f1_score'] * 100:.2f}%")

    st.subheader("Per-Class Precision, Recall & F1-Score")
    rows = []
    for cls_name, metrics in eval_results["per_class_metrics"].items():
        rows.append(
            {
                "Class": format_class_label(cls_name),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1-Score": round(metrics["f1_score"], 4),
                "Support": metrics["support"],
            }
        )
    rows.append(
        {
            "Class": "Macro Average",
            "Precision": round(macro["precision"], 4),
            "Recall": round(macro["recall"], 4),
            "F1-Score": round(macro["f1_score"], 4),
            "Support": eval_results["test_samples"],
        }
    )
    rows.append(
        {
            "Class": "Weighted Average",
            "Precision": round(weighted["precision"], 4),
            "Recall": round(weighted["recall"], 4),
            "F1-Score": round(weighted["f1_score"], 4),
            "Support": eval_results["test_samples"],
        }
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Confusion Matrix")
    if config.CONFUSION_MATRIX_FIG.exists():
        st.image(
            str(config.CONFUSION_MATRIX_FIG),
            caption="Confusion matrix computed on all 1,137 held-out test images.",
            width="stretch",
        )
    else:
        st.warning("Confusion matrix image not found.")

    st.subheader("Training and Validation Curves")
    col_acc, col_loss = st.columns(2)
    with col_acc:
        if config.TRAINING_ACCURACY_FIG.exists():
            st.image(
                str(config.TRAINING_ACCURACY_FIG),
                caption="Training vs. Validation Accuracy",
                width="stretch",
            )
    with col_loss:
        if config.TRAINING_LOSS_FIG.exists():
            st.image(
                str(config.TRAINING_LOSS_FIG),
                caption="Training vs. Validation Loss",
                width="stretch",
            )

    if config.CLASS_DISTRIBUTION_FIG.exists():
        with st.expander("View Dataset Split Distribution"):
            st.image(
                str(config.CLASS_DISTRIBUTION_FIG),
                caption="Stratified 70% / 15% / 15% class distribution across splits.",
                width="stretch",
            )


def render_about_section(model_config: dict | None) -> None:
    """Render architecture, dataset, training methodology, limitations, and tech stack."""
    st.header("About AgroVision AI")

    st.subheader("1. MobileNetV2 & Transfer Learning")
    st.markdown(
        """
        - **MobileNetV2 Backbone:** Uses depthwise separable convolutions and inverted residual
          bottleneck blocks to extract rich visual features with low parameter count (~2.26M base
          parameters), making it suitable for standard laptops and edge devices.
        - **Transfer Learning:** The base model is initialized with pretrained **ImageNet**
          weights (`include_top=False`) and frozen during initial training. A custom
          classification head (`GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(5, softmax)`)
          learns tomato leaf disease boundaries using only **6,405** trainable parameters.
        """
    )

    st.subheader("2. PlantVillage Dataset")
    st.markdown(
        """
        - **Scope:** Five tomato leaf categories (`Tomato Bacterial Spot`, `Tomato Early Blight`,
          `Tomato Late Blight`, `Tomato Leaf Mold`, and `Tomato Healthy`) totaling **7,579**
          verified RGB images from `data/raw/PlantVillage/`.
        - **Stratified Split:** Partitioned deterministically (`seed=42`) via a path-based manifest
          into **70% Training (5,305)**, **15% Validation (1,137)**, and **15% Test (1,137)** with
          zero image overlap across splits.
        """
    )

    st.subheader("3. Preprocessing & Training Pipeline")
    st.markdown(
        """
        - **Consistent Preprocessing:** All images are resized to `224×224` RGB and scaled to
          `[-1, 1]` using `tf.keras.applications.mobilenet_v2.preprocess_input` **once** before
          entering the model.
        - **Training-Only Augmentation:** Horizontal flips, slight rotations, zoom, contrast, and
          brightness shifts are applied only to the training split; validation and test images
          remain unmodified.
        - **Imbalance & Optimization:** Trained with Scikit-learn balanced class weights,
          `Adam (lr=0.001)`, `SparseCategoricalCrossentropy`, `batch_size=32`, `ModelCheckpoint`,
          `EarlyStopping`, and `ReduceLROnPlateau` (best checkpoint saved at Epoch 4).
        """
    )

    if model_config:
        with st.expander("Saved Model Configuration Details"):
            st.json(model_config)

    col_lim, col_tech = st.columns(2)
    with col_lim:
        st.subheader("4. Limitations & Future Scope")
        st.markdown(
            """
            **Limitations:**
            - PlantVillage images are photographed on uniform laboratory backgrounds; accuracy may
              drop on cluttered real-world field photos or extreme lighting.
            - Currently covers 5 tomato leaf classes; unrelated crops or unseen diseases will be
              forced into one of these 5 labels.

            **Future Scope:**
            - Fine-tune top MobileNetV2 layers to improve separation between visually similar
              lesions (`Tomato Early Blight` vs. `Tomato Late Blight`).
            - Expand to all PlantVillage tomato and multi-crop categories plus real field datasets.
            - Export quantized TensorFlow Lite models for offline mobile diagnosis.
            """
        )
    with col_tech:
        st.subheader("5. Technology Stack")
        st.markdown(
            """
            - **Language:** Python 3.12
            - **Deep Learning:** TensorFlow / Keras (MobileNetV2)
            - **Data & Metrics:** NumPy, Pillow, Scikit-learn
            - **Visualization:** Matplotlib, Seaborn
            - **Web Interface:** Streamlit
            """
        )

    st.markdown("")
    render_disclaimer()


def main() -> None:
    apply_custom_styles()

    eval_results = load_json_report(str(config.EVALUATION_RESULTS_PATH))
    model_config = load_json_report(str(config.MODEL_CONFIG_PATH))

    with st.sidebar:
        st.title("🌿 AgroVision AI")
        st.caption("Smart Crop Disease Detection System")
        page = st.radio(
            "Navigation",
            options=["Home", "Disease Detection", "Model Performance", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Model:** `crop_disease_model.keras`")
        if eval_results:
            st.markdown(
                f"**Test Accuracy:** `{eval_results['test_accuracy'] * 100:.2f}%` "
                f"(`N={eval_results['test_samples']:,}`)"
            )
        st.caption("3rd-Year CSE — NNDL Practical Project")

    if page == "Home":
        render_home_section(eval_results, model_config)
    elif page == "Disease Detection":
        render_detection_section()
    elif page == "Model Performance":
        render_performance_section(eval_results)
    elif page == "About":
        render_about_section(model_config)


if __name__ == "__main__":
    main()
