import cv2
import gradio as gr
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_faces(image):
    if image is None:
        return None, "No image provided."

    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 200, 100), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_bgr[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 100, 0), 1)
        label = f"Face {faces.tolist().index([x, y, w, h]) + 1}"
        cv2.putText(img_bgr, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)

    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    count = len(faces)
    summary = f"Detected {count} face{'s' if count != 1 else ''}."
    return result, summary


demo = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="numpy", label="Result"),
        gr.Textbox(label="Summary"),
    ],
    title="Face Detection — Vasu Manocha",
    description=(
        "Real-time face and eye detection using OpenCV Haar Cascades. "
        "Originally built as an automated attendance system. Upload any photo to try it."
    ),
    examples=[],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
