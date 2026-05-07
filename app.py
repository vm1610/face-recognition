import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection Demo", page_icon="👤")

st.title("👤 Face Detection")
st.caption("Upload a photo — OpenCV detects faces and eyes in real time. Built by [Vasu Manocha](https://vasumanocha.com).")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img   = np.array(image)
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 100), 2)
        roi_gray  = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 100, 0), 1)
        cv2.putText(img, f"Face {i + 1}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

    count = len(faces)
    st.image(img, use_container_width=True)
    if count == 0:
        st.warning("No faces detected — try a clearer photo.")
    else:
        st.success(f"Detected {count} face{'s' if count != 1 else ''}!")
