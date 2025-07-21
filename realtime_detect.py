import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import time
import os
from collections import deque, Counter
import matplotlib.pyplot as plt
from datetime import datetime

# Load trained model
model = tf.keras.models.load_model("model/mobilenet_model.h5")
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['space', 'del', 'nothing']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize TTS engine safely
try:
    tts = pyttsx3.init()
except Exception as e:
    tts = None
    print(f"TTS initialization failed: {e}")

# Buffer and speech setup
prediction_queue = deque(maxlen=10)
predicted_sentence = ""
last_spoken_time = time.time()
speak_delay = 5

# Directory for screenshots
os.makedirs("screenshots", exist_ok=True)
transcript_path = "transcript.txt"

# Live plot setup
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
bar_plot = ax.barh(range(5), [0] * 5, color='cyan')
text_labels = [ax.text(0, i, "", va='center') for i in range(5)]
ax.set_xlim(0, 1)
ax.set_yticks(range(5))
ax.set_yticklabels([''] * 5)
ax.set_xlabel('Confidence')
fig.canvas.manager.set_window_title('Top 5 Predictions')

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    most_likely = "nothing"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = max(int(min(x_coords) * w) - 20, 0)
            y_min = max(int(min(y_coords) * h) - 20, 0)
            x_max = min(int(max(x_coords) * w) + 20, w)
            y_max = min(int(max(y_coords) * h) + 20, h)

            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                continue

            resized = cv2.resize(cropped, (64, 64))
            input_tensor = np.expand_dims(resized, axis=0).astype('float32') / 255.0
            prediction = model.predict(input_tensor, verbose=0)[0]

            top5_idx = prediction.argsort()[-5:][::-1]
            top5_preds = [(class_names[i], prediction[i]) for i in top5_idx]
            most_likely = top5_preds[0][0]
            prediction_queue.append(most_likely)

            # Draw prediction bars on frame
            for i, (label, score) in enumerate(top5_preds):
                bar_length = int(score * 200)
                y_pos = 50 + i * 30
                cv2.rectangle(frame, (10, y_pos), (10 + bar_length, y_pos + 20), (0, 255, 0), -1)
                cv2.putText(frame, f"{label}: {int(score * 100)}%", (220, y_pos + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update matplotlib plot
            for j, (label, score) in enumerate(top5_preds):
                bar_plot[j].set_width(score)
                text_labels[j].set_text(f"{label} ({int(score * 100)}%)")
            ax.set_yticklabels([label for label, _ in top5_preds])
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Sentence construction
    if prediction_queue:
        most_common = Counter(prediction_queue).most_common(1)[0][0]

        if most_common == "space":
            if not predicted_sentence.endswith(" "):
                predicted_sentence += " "
        elif most_common == "del":
            predicted_sentence = predicted_sentence[:-1]
        elif most_common == "S":
            filename = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            cv2.putText(frame, "📸 Screenshot Saved!", (10, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif most_common != "nothing":
            if len(predicted_sentence) == 0 or predicted_sentence[-1] != most_common:
                predicted_sentence += most_common

        # Speak the sentence
        if time.time() - last_spoken_time > speak_delay and predicted_sentence.strip():
            if tts:
                tts.say(predicted_sentence)
                tts.runAndWait()
            last_spoken_time = time.time()

        # Save transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(predicted_sentence)

    # Subtitle overlay
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Subtitles: {predicted_sentence}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Sign Language Recognition - Advanced", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
