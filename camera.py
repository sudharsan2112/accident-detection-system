import cv2
import numpy as np
import datetime
import time
import threading

from detection import AccidentDetectionModel
from telegram_alert import send_full_alert, send_telegram_message
from config import (
    VIDEO_SOURCE,
    MODEL_JSON,
    MODEL_WEIGHTS,
    CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE,
    COOLDOWN_SECONDS,
    FRAME_SKIP,
)

# Load the model once at startup
print("[INFO] Loading accident detection model...")
model = AccidentDetectionModel(
    model_json_file=MODEL_JSON,
    model_weights_file=MODEL_WEIGHTS
)
print("[INFO] Model loaded successfully.")

font = cv2.FONT_HERSHEY_SIMPLEX


def trigger_alert(frame, accident_prob):
    """
    Send the appropriate alert based on confidence level.
    Runs in a background thread so the video loop never freezes.
    """
    if accident_prob >= HIGH_CONFIDENCE * 100:
        # High confidence → text + photo
        print(f"[ALERT] High confidence ({accident_prob}%) — sending photo + message")
        send_full_alert(frame, accident_prob)
    else:
        # Medium confidence → text only
        print(f"[ALERT] Medium confidence ({accident_prob}%) — sending message only")
        send_telegram_message(accident_prob)


def startapplication():
    print(f"[INFO] Opening video source: {VIDEO_SOURCE}")
    video = cv2.VideoCapture(VIDEO_SOURCE)

    if not video.isOpened():
        print("[ERROR] Cannot open video source. Check VIDEO_SOURCE in config.py")
        return

    last_alert_time = 0
    frame_count = 0

    print("[INFO] Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = video.read()

        # FIXED: Distinguish end-of-file from a real read error
        if not ret:
            print("[INFO] Video ended or frame could not be read. Exiting.")
            break

        frame_count += 1

        # FIXED: Skip frames for performance (process every Nth frame)
        if frame_count % FRAME_SKIP != 0:
            cv2.imshow('Accident Detection System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Preprocess frame for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250)).astype("float32") / 255.0

        # Run prediction
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        # FIXED: Always use explicit index for correct probability
        accident_prob = round(prob[0][0] * 100, 2)

        # Display result on frame
        if pred == "Accident":
            color = (0, 0, 255)   # Red for accident
            label = f"ACCIDENT {accident_prob}%"
        else:
            color = (0, 255, 0)   # Green for no accident
            no_accident_prob = round(prob[0][1] * 100, 2)
            label = f"No Accident {no_accident_prob}%"

        cv2.rectangle(frame, (0, 0), (380, 45), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 30), font, 1, color, 2)

        # Timestamp on frame
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                    font, 0.5, (255, 255, 255), 1)

        # FIXED: Confidence threshold + cooldown + background thread
        current_time = time.time()
        if (pred == "Accident"
                and prob[0][0] >= CONFIDENCE_THRESHOLD
                and current_time - last_alert_time > COOLDOWN_SECONDS):

            # FIXED: Use frame.copy() — frame changes each loop iteration
            threading.Thread(
                target=trigger_alert,
                args=(frame.copy(), accident_prob),
                daemon=True
            ).start()

            last_alert_time = current_time

        cv2.imshow('Accident Detection System', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            print("[INFO] Quit key pressed.")
            break

    video.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    startapplication()
