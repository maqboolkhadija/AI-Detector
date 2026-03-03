# =====================================
# SMART AI OBJECT DETECTOR 🚀
# =====================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from groq import Groq

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="🔥 Smart AI Detector",
    page_icon="🚀",
    layout="wide"
)

st.title("🔥 Smart AI Object Detector")
st.markdown("Camera ke samne koi bhi cheez dikhao — AI uska **name + use** batayega 😎")

st.divider()

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# -------------------------------
# ADD YOUR GROQ API KEY HERE
# -------------------------------

os.environ["GROQ_API_KEY"] = "APNA_GROQ_API_KEY_YAHAN_DALO"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# -------------------------------
# FUNCTION: GET OBJECT USE
# -------------------------------

def get_object_use(object_name):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"What is the use of {object_name}? Explain in 2 simple lines.",
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content

    except Exception:
        return "Explanation not available."

# -------------------------------
# FUNCTION: DETECT OBJECTS
# -------------------------------

def detect_objects(frame):

    results = model.predict(frame, conf=0.5)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            detected_objects.append(label)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Write label
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

    return frame, list(set(detected_objects))

# -------------------------------
# CAMERA INPUT
# -------------------------------

st.subheader("📷 Camera Input")

camera_image = st.camera_input("Show something to AI")

if camera_image is not None:

    image = Image.open(camera_image)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    output_frame, objects = detect_objects(frame)
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    st.image(output_frame, caption="Detected Objects", use_column_width=True)

    st.divider()

    if objects:
        st.subheader("🧠 AI Explanation")

        for obj in objects:
            st.markdown(f"### 🔎 {obj.capitalize()}")
            explanation = get_object_use(obj)
            st.write(explanation)
    else:
        st.warning("No object detected. Try again.")
