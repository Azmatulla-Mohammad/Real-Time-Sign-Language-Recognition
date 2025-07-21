import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time
from collections import deque
from datetime import datetime
from PIL import Image
import tempfile
import os

# ---- Init Session State ---- #
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = time.time()

# ---- Load Model ---- #
model = tf.keras.models.load_model("model/mobilenet_model.h5")
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'
]

# ---- Text-to-Speech ---- #
tts = pyttsx3.init()
speak_delay = 5  # seconds

# ---- Streamlit UI ---- #
st.set_page_config(page_title="Webcam Sign Language Recognizer", layout="wide")
st.title("🤟 Real-Time Sign Language Recognition with Streamlit")

# ---- Sidebar ---- #
st.sidebar.header("Instructions")
st.sidebar.markdown("""
- Use webcam to detect hand signs.
- Top 5 predictions are shown.
- `S` triggers a screenshot.
- Final sentence will be spoken aloud.
""")
speak_enabled = st.sidebar.toggle("🔊 Enable Text-to-Speech", value=True)
auto_clear = st.sidebar.button("🧹 Clear Sentence")

# ---- Webcam Feed ---- #
FRAME_WINDOW = st.image([])
prediction_queue = deque(maxlen=10)

# ---- OpenCV Webcam ---- #
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize input for model
    img = cv2.resize(frame, (64, 64))
    input_tensor = np.expand_dims(img, axis=0).astype("float32") / 255.0

    # Predict
    prediction = model.predict(input_tensor, verbose=0)[0]
    top5_idx = prediction.argsort()[-5:][::-1]
    top5_preds = [(class_names[i], prediction[i]) for i in top5_idx]
    top1 = top5_preds[0][0]
    prediction_queue.append(top1)

    # Add to sentence
    most_common = max(set(prediction_queue), key=prediction_queue.count)
    if most_common == "space":
        if not st.session_state.sentence.endswith(" "):
            st.session_state.sentence += " "
    elif most_common == "del":
        st.session_state.sentence = st.session_state.sentence[:-1]
    elif most_common != "nothing":
        if len(st.session_state.sentence) == 0 or st.session_state.sentence[-1] != most_common:
            st.session_state.sentence += most_common

    # Speak
    if speak_enabled and time.time() - st.session_state.last_spoken > speak_delay:
        if st.session_state.sentence.strip():
            tts.say(st.session_state.sentence)
            tts.runAndWait()
            st.session_state.last_spoken = time.time()

    # Screenshot on key "S"
    if cv2.waitKey(1) & 0xFF == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshot_{timestamp}.png"
        cv2.imwrite(path, frame)
        st.success(f"📸 Screenshot saved as `{path}`")

    # Overlay predictions
    for i, (label, score) in enumerate(top5_preds):
        cv2.putText(frame, f"{label}: {int(score * 100)}%", (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Sentence display
    cv2.putText(frame, f"Sentence: {st.session_state.sentence}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # Break if clear or stop
    if auto_clear:
        st.session_state.sentence = ""
        st.rerun()

# Cleanup
cap.release()
cv2.destroyAllWindows()
