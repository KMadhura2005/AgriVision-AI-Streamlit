import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "agri_vision_model_9class.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

BACKGROUND_PATH = os.path.join(
    BASE_DIR,
    "agri-background-image.png"
)


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found:\n\n"
        + MODEL_PATH
    )
    st.stop()

if not os.path.exists(CLASS_NAMES_PATH):
    st.error(
        "❌ class_names.json not found:\n\n"
        + CLASS_NAMES_PATH
    )
    st.stop()

if not os.path.exists(BACKGROUND_PATH):
    st.error(
        "❌ Background image not found:\n\n"
        + BACKGROUND_PATH
    )
    st.stop()


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

with open(
    BACKGROUND_PATH,
    "rb"
) as image_file:

    background_base64 = base64.b64encode(
        image_file.read()
    ).decode()


# =========================================================
# BACKGROUND CSS
# =========================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image:
            url(
                "data:image/png;base64,{background_base64}"
            );

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "camera_on" not in st.session_state:

    st.session_state.camera_on = False


if "image_reset" not in st.session_state:

    st.session_state.image_reset = 0


if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================================================
# LOAD CLASS NAMES
# =========================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# Load everything

model = load_model()

class_names = load_class_names()


# =========================================================
# FORMAT CLASS NAME
# =========================================================

def format_class_name(class_name):

    name = class_name.replace(
        "___",
        " - "
    )

    name = name.replace(
        "__",
        " - "
    )

    name = name.replace(
        "_",
        " "
    )

    return name


# =========================================================
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {

        "crop": "🌶️ Pepper",

        "disease": "Bacterial Spot",

        "symptoms":
            "Small dark spots may appear on leaves and fruit.",

        "treatment":
            "Remove severely infected leaves and maintain good field hygiene.",

        "prevention":
            "Avoid overhead watering and provide good air circulation."
    },


    "Pepper__bell___healthy": {

        "crop": "🌶️ Pepper",

        "disease": "Healthy",

        "symptoms":
            "No major visible signs of disease.",

        "treatment":
            "No disease treatment is required.",

        "prevention":
            "Continue proper watering, nutrition, and regular monitoring."
    },


    "Potato___Early_blight": {

        "crop": "🥔 Potato",

        "disease": "Early Blight",

        "symptoms":
            "Dark spots with concentric rings may appear on older leaves.",

        "treatment":
            "Remove affected leaves and maintain proper plant spacing and airflow.",

        "prevention":
            "Avoid prolonged leaf wetness and remove infected plant debris."
    },


    "Potato___Late_blight": {

        "crop": "🥔 Potato",

        "disease": "Late Blight",

        "symptoms":
            "Dark water-soaked lesions can develop on leaves and stems.",

        "treatment":
            "Remove infected plant material and seek appropriate agricultural guidance.",

        "prevention":
            "Reduce excess moisture and avoid prolonged wet conditions."
    },


    "Potato___healthy": {

        "crop": "🥔 Potato",

        "disease": "Healthy",

        "symptoms":
            "No major visible signs of disease.",

        "treatment":
            "No disease treatment is required.",

        "prevention":
            "Maintain proper watering, nutrition, and regular crop monitoring."
    },


    "Tomato_Early_blight": {

        "crop": "🍅 Tomato",

        "disease": "Early Blight",

        "symptoms":
            "Brown spots with concentric rings commonly appear on older leaves.",

        "treatment":
            "Remove affected leaves and improve airflow around the plants.",

        "prevention":
            "Avoid overhead watering and remove infected plant debris."
    },


    "Tomato_Late_blight": {

        "crop": "🍅 Tomato",

        "disease": "Late Blight",

        "symptoms":
            "Dark, water-soaked lesions may develop on leaves and stems.",

        "treatment":
            "Remove infected plant material and seek appropriate agricultural guidance.",

        "prevention":
            "Reduce excess moisture and avoid prolonged wet conditions."
    },


    "Tomato_Leaf_Mold": {

        "crop": "🍅 Tomato",

        "disease": "Leaf Mold",

        "symptoms":
            "Yellowish areas may appear on the upper leaf surface with mold growth underneath.",

        "treatment":
            "Remove severely affected leaves and improve ventilation.",

        "prevention":
            "Reduce humidity and provide sufficient spacing and airflow."
    },


    "Tomato_healthy": {

        "crop": "🍅 Tomato",

        "disease": "Healthy",

        "symptoms":
            "No major visible signs of disease.",

        "treatment":
            "No disease treatment is required.",

        "prevention":
            "Continue proper watering, nutrition, and regular monitoring."
    }
}


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_leaf(image):

    # Open image

    img = Image.open(
        image
    ).convert("RGB")


    # Resize

    img = img.resize(
        (128, 128)
    )


    # Convert to NumPy

    img_array = np.array(
        img
    )


    # Normalize

    img_array = img_array.astype(
        "float32"
    ) / 255.0


    # Add batch dimension

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Prediction

    predictions = model.predict(
        img_array,
        verbose=0
    )


    probabilities = predictions[0]


    # Top 2 predictions

    top_indices = np.argsort(
        probabilities
    )[::-1][:2]


    # Primary prediction

    first_index = top_indices[0]

    first_class = class_names[
        first_index
    ]

    first_confidence = (
        float(
            probabilities[first_index]
        ) * 100
    )


    # Alternative prediction

    second_index = top_indices[1]

    second_class = class_names[
        second_index
    ]

    second_confidence = (
        float(
            probabilities[second_index]
        ) * 100
    )


    return (
        first_class,
        first_confidence,
        second_class,
        second_confidence
    )


# =========================================================
# NAVBAR
# =========================================================

nav1, nav2 = st.columns(
    [1, 1]
)

with nav1:

    st.title(
        "🌱 AgriVision AI"
    )

with nav2:

    st.markdown(
        """
        <div style="
            text-align:right;
            padding-top:25px;
            font-weight:600;
            color:#315d3e;
        ">
            AI-Based Crop Disease Detection
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# HERO SECTION
# =========================================================

left, right = st.columns(
    [1.15, 0.85],
    gap="large"
)


# =========================================================
# LEFT HERO
# =========================================================

with left:

    st.markdown(
        "🤖 **AI-POWERED CROP HEALTH ANALYSIS**"
    )

    st.header(
        "Detect Crop Diseases"
    )

    st.subheader(
        "Instantly 🌱"
    )

    st.write(
        """
        Upload a clear leaf image of tomato,
        potato, or pepper and let AgriVision AI
        analyze it using a trained deep learning model.
        """
    )

    st.write(
        "🧠 CNN Model   |   🌱 3 Crops   |   🔬 9 Conditions"
    )


# =========================================================
# RIGHT HERO
# =========================================================

with right:

    st.subheader(
        "🌿 Crop Health Detection"
    )

    st.info(
        """
        Upload a leaf image or take a photo using
        your camera to get an AI-powered crop
        health prediction.
        """
    )

    st.write(
        "🍅 Tomato   |   🥔 Potato   |   🌶️ Pepper"
    )


# =========================================================
# INPUT SECTION
# =========================================================

st.divider()

st.subheader(
    "📷 Analyze Your Crop Leaf"
)


upload_col, camera_col = st.columns(
    2,
    gap="large"
)


# =========================================================
# UPLOAD
# =========================================================

with upload_col:

    st.markdown(
        "### 📁 Upload an Image"
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key=(
            "upload_"
            + str(
                st.session_state.image_reset
            )
        )
    )


# =========================================================
# CAMERA
# =========================================================

camera_image = None


with camera_col:

    st.markdown(
        "### 📷 Take a Photo"
    )


    if not st.session_state.camera_on:

        turn_on_camera = st.button(
            "📷 Turn On Camera",
            use_container_width=True
        )


        if turn_on_camera:

            st.session_state.camera_on = True

            st.rerun()


    else:

        turn_off_camera = st.button(
            "❌ Turn Off Camera",
            use_container_width=True
        )


        if turn_off_camera:

            st.session_state.camera_on = False

            st.session_state.image_reset += 1

            st.rerun()


        camera_image = st.camera_input(
            "Capture a leaf image",
            key=(
                "camera_"
                + str(
                    st.session_state.image_reset
                )
            )
        )


# =========================================================
# SELECT IMAGE
# =========================================================

image_source = None


if camera_image is not None:

    image_source = camera_image

elif uploaded_file is not None:

    image_source = uploaded_file


# =========================================================
# IMAGE PREVIEW
# =========================================================

if image_source is not None:

    st.divider()

    st.subheader(
        "🖼️ Selected Leaf"
    )


    preview_col1, preview_col2, preview_col3 = st.columns(
        [1, 2, 1]
    )


    with preview_col2:

        st.image(
            image_source,
            caption="Leaf image ready for analysis",
            use_container_width=True
        )


    st.success(
        "Leaf image captured successfully! 🌱"
    )


    # =====================================================
    # ACTION BUTTONS
    # =====================================================

    button1, button2, button3 = st.columns(
        [1, 1, 1]
    )


    with button2:

        analyze = st.button(
            "🔍 Analyze Leaf",
            use_container_width=True,
            type="primary"
        )


    with button3:

        delete_image = st.button(
            "🗑️ Delete Image",
            use_container_width=True
        )


    # =====================================================
    # DELETE IMAGE
    # =====================================================

    if delete_image:

        st.session_state.image_reset += 1

        st.session_state.camera_on = False

        st.rerun()


    # =====================================================
    # ANALYZE
    # =====================================================

    if analyze:

        with st.spinner(
            "🤖 AgriVision AI is analyzing the leaf..."
        ):

            try:

                (
                    prediction,
                    confidence,
                    second_prediction,
                    second_confidence
                ) = predict_leaf(
                    image_source
                )


                # =============================================
                # FORMAT
                # =============================================

                prediction_display = format_class_name(
                    prediction
                )

                second_prediction_display = format_class_name(
                    second_prediction
                )


                # =============================================
                # SAVE HISTORY
                # =============================================

                st.session_state.prediction_history.append(
                    {
                        "Prediction":
                            prediction_display,

                        "Confidence":
                            f"{confidence:.2f}%",

                        "Alternative":
                            second_prediction_display
                    }
                )


                # =============================================
                # RESULT
                # =============================================

                st.divider()

                st.header(
                    "🤖 AI Analysis Result"
                )


                result_col1, result_col2 = st.columns(
                    2,
                    gap="large"
                )


                with result_col1:

                    st.metric(
                        "🌱 Prediction",
                        prediction_display
                    )


                with result_col2:

                    st.metric(
                        "📊 AI Confidence",
                        f"{confidence:.2f}%"
                    )


                # =============================================
                # CONFIDENCE BAR
                # =============================================

                st.write(
                    "Confidence Level"
                )

                st.progress(
                    min(
                        max(
                            confidence / 100,
                            0.0
                        ),
                        1.0
                    )
                )


                # =============================================
                # ALTERNATIVE
                # =============================================

                st.info(
                    "🔎 Alternative Prediction: "
                    f"**{second_prediction_display}** "
                    f"({second_confidence:.2f}%)"
                )


                # =============================================
                # CONFIDENCE MESSAGE
                # =============================================

                st.subheader(
                    "📈 Confidence Assessment"
                )


                if confidence >= 90:

                    st.success(
                        "🟢 High Confidence — "
                        "The model is highly confident "
                        "in this prediction."
                    )

                elif confidence >= 70:

                    st.warning(
                        "🟡 Moderate Confidence — "
                        "Try uploading a clearer leaf "
                        "image for better accuracy."
                    )

                else:

                    st.error(
                        "🔴 Low Confidence — "
                        "Please capture or upload a "
                        "clearer leaf image."
                    )


                # =============================================
                # RECOMMENDATION
                # =============================================

                st.subheader(
                    "💡 AI Recommendation"
                )


                if "healthy" in prediction.lower():

                    st.success(
                        "🌱 The leaf appears healthy. "
                        "Continue regular watering, "
                        "proper nutrition, and regular monitoring."
                    )


                elif "early_blight" in prediction.lower():

                    st.warning(
                        "🦠 Early blight detected. "
                        "Remove affected leaves, avoid "
                        "overhead watering, and maintain "
                        "good airflow around the plant."
                    )


                elif "late_blight" in prediction.lower():

                    st.warning(
                        "🦠 Late blight detected. "
                        "Remove infected plant material "
                        "and reduce excess moisture. "
                        "Consider agricultural expert guidance."
                    )


                elif "leaf_mold" in prediction.lower():

                    st.warning(
                        "🦠 Leaf mold detected. "
                        "Improve ventilation, reduce humidity, "
                        "and remove severely affected leaves."
                    )


                elif "bacterial_spot" in prediction.lower():

                    st.warning(
                        "🦠 Bacterial spot detected. "
                        "Remove infected leaves, avoid "
                        "splashing water onto foliage, "
                        "and maintain good field hygiene."
                    )


                else:

                    st.info(
                        "🌿 A crop health issue has been "
                        "detected. Consider consulting "
                        "an agricultural expert."
                    )


                # =============================================
                # DETAILED INFORMATION
                # =============================================

                info = DISEASE_INFO.get(
                    prediction
                )


                if info is not None:

                    st.divider()

                    st.subheader(
                        "📋 Detailed Crop Health Information"
                    )


                    info_col1, info_col2 = st.columns(
                        2,
                        gap="large"
                    )


                    with info_col1:

                        st.write(
                            "🌱 **Crop**"
                        )

                        st.info(
                            info["crop"]
                        )


                        st.write(
                            "🦠 **Condition**"
                        )

                        st.success(
                            info["disease"]
                        )


                        st.write(
                            "🔍 **Common Symptoms**"
                        )

                        st.write(
                            info["symptoms"]
                        )


                    with info_col2:

                        st.write(
                            "💊 **Recommended Action**"
                        )

                        st.warning(
                            info["treatment"]
                        )


                        st.write(
                            "🛡️ **Prevention**"
                        )

                        st.write(
                            info["prevention"]
                        )


            except Exception as e:

                st.error(
                    "❌ Prediction failed"
                )

                st.exception(
                    e
                )


# =========================================================
# PREDICTION HISTORY
# =========================================================

if len(
    st.session_state.prediction_history
) > 0:

    st.divider()

    st.header(
        "📜 Prediction History"
    )


    clear_col1, clear_col2 = st.columns(
        [4, 1]
    )


    with clear_col1:

        st.caption(
            "Previous analyses from this session"
        )


    with clear_col2:

        clear_history = st.button(
            "🗑️ Clear History",
            use_container_width=True
        )


    if clear_history:

        st.session_state.prediction_history = []

        st.rerun()


    for number, result in enumerate(
        reversed(
            st.session_state.prediction_history
        ),
        start=1
    ):

        with st.expander(
            f"🔎 Analysis {number} — "
            f"{result['Prediction']}"
        ):

            history_col1, history_col2 = st.columns(
                2
            )


            with history_col1:

                st.write(
                    "🌱 **Prediction:** "
                    + result["Prediction"]
                )

                st.write(
                    "📊 **Confidence:** "
                    + result["Confidence"]
                )


            with history_col2:

                st.write(
                    "🔎 **Alternative:** "
                    + result["Alternative"]
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    "🌱 **AgriVision AI — AI-Based Crop Disease Detection System**"
)

st.caption(
    "Smart • Fast • AI-Powered Crop Health Analysis"
)