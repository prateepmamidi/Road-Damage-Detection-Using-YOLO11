# ==========================================================
# ROAD DAMAGE DETECTION SYSTEM
# Streamlit + YOLO11n
# ==========================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd
from collections import Counter

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

MODEL_PATH = "C:/Users/SRMAP_JC203/Desktop/prateep/vizag_intern/archive/dataset/runs/detect/RoadDamage_Project/YOLO11n_Final/weights/best.pt"
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

CLASS_NAMES = model.names

# ==========================================================
# APPLICATION HEADER
# ==========================================================

st.title("🛣️ Road Damage Detection System")

st.markdown("""
Upload a road image to automatically detect:

- 🕳️ Potholes
- 🪨 Cracks
- ⭕ Manholes
""")
st.sidebar.header("Detection Settings")

confidence = st.sidebar.number_input(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================================
# IMAGE UPLOAD AND OBJECT DETECTION
# ==========================================================

if uploaded_file is not None:

    # Display Uploaded Image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_image_path = tmp_file.name

    # Run YOLO Prediction
    results = model.predict(
        source=temp_image_path,
        conf=confidence,
        save=False,
        verbose=False
    )

    # Annotated Image
    annotated = results[0].plot()

    with col2:
        st.subheader("Detection Result")
        st.image(annotated, channels="BGR", use_container_width=True)

    # ==========================================================
    # DETECTION SUMMARY
    # ==========================================================

    st.markdown("---")
    st.subheader("Detection Summary")

    counter = Counter()
    detection_data = []

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = CLASS_NAMES[cls]

        counter[class_name] += 1

        detection_data.append({
            "Class": class_name.title(),
            "Confidence": f"{conf*100:.2f}%"
        })

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🕳️ Potholes",
            counter.get("pothole", 0)
        )

    with col2:
        st.metric(
            "🪨 Cracks",
            counter.get("crack", 0)
        )

    with col3:
        st.metric(
            "⭕ Manholes",
            counter.get("manhole", 0)
        )

    with col4:
        st.metric(
            "📦 Total Objects",
            sum(counter.values())
        )

    # ==========================================================
    # DETECTION TABLE
    # ==========================================================

    st.markdown("---")
    st.subheader("Detected Objects")

    if len(detection_data) > 0:

        df = pd.DataFrame(detection_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("No road damage detected.")

    # ==========================================================
    # DOWNLOAD RESULT IMAGE
    # ==========================================================

    result_image = Image.fromarray(annotated)

    output_path = "prediction_result.jpg"

    result_image.save(output_path)

    with open(output_path, "rb") as file:

        st.download_button(
            label="⬇️ Download Detection Result",
            data=file,
            file_name="Road_Damage_Result.jpg",
            mime="image/jpeg"
        )

    # Remove temporary file
    os.remove(temp_image_path)




    