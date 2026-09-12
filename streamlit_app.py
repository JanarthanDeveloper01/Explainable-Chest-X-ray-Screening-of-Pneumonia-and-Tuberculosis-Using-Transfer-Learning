import sys
print("=" * 50)
print("Python Executable:")
print(sys.executable)
print("=" * 50)

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import time


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Pneumonia & TB Detection",
    page_icon="🫁",
    layout="centered"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_my_model():
    return load_model(
        "Final_Pneumonia_TB_ResNet50.keras"
    )


model = load_my_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Normal",
    "Pneumonia",
    "Tuberculosis"
]


# =========================================================
# EXPLAINABLE AI - GRAD-CAM
# =========================================================

def generate_gradcam(model, img_array, predicted_index):

    last_conv_layer_name = "conv5_block3_out"

    last_conv_layer = model.get_layer(
        last_conv_layer_name
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) > 0:

        heatmap = heatmap / max_value

    return heatmap.numpy()


# =========================================================
# CREATE GRAD-CAM OVERLAY
# =========================================================

def create_gradcam_overlay(original_image, heatmap):

    original_array = np.array(
        original_image.convert("RGB")
    )

    original_height = original_array.shape[0]
    original_width = original_array.shape[1]

    heatmap = cv2.resize(
        heatmap,
        (original_width, original_height)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_array,
        0.60,
        colored_heatmap,
        0.40,
        0
    )

    return overlay


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }

    .sub-text {
        text-align: center;
        color: gray;
        font-size: 16px;
    }

    .result-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<p class="main-title">🫁 AI Chest X-Ray Detection</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="sub-text">
    Upload a Chest X-ray to detect Normal,
    Pneumonia, or Tuberculosis
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# PREDICTION STARTS
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # LOAD ORIGINAL IMAGE
    # =====================================================

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = original_image.resize(
        (224, 224)
    )

    st.image(
        original_image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )


    # =====================================================
    # IMAGE PREPROCESSING
    # =====================================================

    img_array = np.array(
        image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)


    # =====================================================
    # ANALYSIS MESSAGE
    # =====================================================

    st.write(
        "🔎 Analyzing image..."
    )

    time.sleep(1)


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    confidence = float(
        np.max(predictions)
    )

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]


    # =====================================================
    # PROBABILITY SCORES
    # =====================================================

    normal_prob = float(
        predictions[0] * 100
    )

    pneumonia_prob = float(
        predictions[1] * 100
    )

    tb_prob = float(
        predictions[2] * 100
    )


    # =====================================================
    # AI PREDICTION RESULT
    # =====================================================

    st.divider()

    st.subheader(
        "🧪 AI Prediction Result"
    )


    if predicted_class == "Normal":

        st.success(
            f"Prediction: {predicted_class}"
        )

        severity = "Very Low Risk"

        recommendation = """
        No major disease-related abnormalities were detected
        in the chest X-ray.

        Patient condition appears normal based on AI analysis.
        """

        feedback = """
        The AI model found image patterns with high similarity
        to the normal chest X-ray images learned during training.
        """


    elif predicted_class == "Pneumonia":

        st.error(
            f"Prediction: {predicted_class}"
        )

        if confidence > 0.85:

            severity = "High Severity"

        else:

            severity = "Moderate Severity"

        recommendation = """
        Pneumonia-related image patterns were detected.

        Medical consultation and clinical verification are
        recommended.
        """

        feedback = """
        The AI model detected image patterns similar to
        pneumonia examples learned during model training.
        """


    elif predicted_class == "Tuberculosis":

        st.error(
            f"Prediction: {predicted_class}"
        )

        if confidence > 0.85:

            severity = "Critical"

        else:

            severity = "High Risk"

        recommendation = """
        Tuberculosis-related image patterns were detected.

        TB screening and professional medical evaluation are
        strongly recommended.
        """

        feedback = """
        The AI model identified image patterns similar to
        tuberculosis chest X-ray examples learned during training.
        """


    # =====================================================
    # CONFIDENCE ANALYSIS
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Confidence Analysis"
    )

    st.progress(
        int(confidence * 100)
    )

    st.write(
        f"### Overall Confidence Score: "
        f"{confidence * 100:.2f}%"
    )

    st.info(
        """
        The confidence score represents the model's probability
        for the selected prediction.

        A high softmax confidence score does not by itself prove
        diagnostic certainty or clinical severity.
        """
    )


    # =====================================================
    # DISEASE PROBABILITY BREAKDOWN
    # =====================================================

    st.subheader(
        "📈 Disease Probability Breakdown"
    )

    st.write(
        f"🟢 Normal: {normal_prob:.2f}%"
    )

    st.write(
        f"🟠 Pneumonia: {pneumonia_prob:.2f}%"
    )

    st.write(
        f"🔴 Tuberculosis: {tb_prob:.2f}%"
    )


    # =====================================================
    # EXPLAINABLE AI ANALYSIS - GRAD-CAM
    # =====================================================

    st.divider()

    st.subheader(
        "🧠 Explainable AI Analysis"
    )

    st.write(
        """
        Grad-CAM highlights image regions that had a stronger
        influence on the AI model's selected prediction.
        """
    )


    try:

        with st.spinner(
            "Generating AI attention heatmap..."
        ):

            heatmap = generate_gradcam(
                model,
                img_array,
                predicted_index
            )

            gradcam_overlay = create_gradcam_overlay(
                original_image,
                heatmap
            )


        # =================================================
        # DISPLAY ORIGINAL AND GRAD-CAM
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                original_image,
                caption="Original Chest X-ray",
                use_container_width=True
            )


        with col2:

            st.image(
                gradcam_overlay,
                caption=(
                    f"Grad-CAM — "
                    f"{predicted_class}"
                ),
                use_container_width=True
            )


        # =================================================
        # GRAD-CAM INTERPRETATION
        # =================================================

        st.info(
            """
            🔍 **How to interpret the attention map**

            🔴 **Red / Orange**
            — Higher Grad-CAM activation

            🟡 **Yellow**
            — Moderate Grad-CAM activation

            🔵 **Blue**
            — Lower Grad-CAM activation
            """
        )


        st.warning(
            """
            ⚠️ **Explainability Note**

            Grad-CAM visualizes regions that influenced the
            model's prediction.

            It does not identify an exact lesion boundary and
            should not be interpreted as medical segmentation.
            """
        )


    except Exception as e:

        st.error(
            "Grad-CAM explanation could not be generated."
        )

        st.write(
            "Technical details:",
            str(e)
        )


    # =====================================================
    # SEVERITY ASSISTANT
    # =====================================================

    st.divider()

    st.subheader(
        "🚨 Severity Assistant"
    )

    st.write(
        f"Severity Level: **{severity}**"
    )


    if severity == "Very Low Risk":

        st.success(
            "Minimal model-indicated disease risk."
        )


    elif severity == "Moderate Severity":

        st.warning(
            "Moderate model confidence for pneumonia."
        )


    elif severity == "High Severity":

        st.error(
            "High model confidence for pneumonia."
        )


    elif severity == "High Risk":

        st.warning(
            "High model confidence for tuberculosis."
        )


    elif severity == "Critical":

        st.error(
            "Very high model confidence for tuberculosis."
        )


    # =====================================================
    # AI FEEDBACK
    # =====================================================

    st.divider()

    st.subheader(
        "🤖 AI Feedback"
    )

    st.info(
        feedback
    )


    # =====================================================
    # CLINICAL RECOMMENDATION
    # =====================================================

    st.subheader(
        "🩺 Recommended Next Step"
    )

    st.warning(
        recommendation
    )


    # =====================================================
    # FINAL DISCLAIMER
    # =====================================================

    st.divider()

    st.warning(
        """
        ⚠️ **Disclaimer**

        This AI system is developed for educational and
        research purposes only.

        The output is a model prediction from a chest X-ray
        image and must not be considered a final medical
        diagnosis.

        The confidence score is not a clinically validated
        disease-severity score.

        Always consult a qualified medical professional and
        use appropriate clinical or laboratory confirmation.
        """
    )