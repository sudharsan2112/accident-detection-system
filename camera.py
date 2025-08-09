import cv2
from detection import AccidentDetectionModel
import numpy as np
import pywhatkit as kit
import datetime
import time  # For cooldown

# Initialize the model for accident detection
model_json = r'model.json'
model_file = r'C:\Users\HP\Downloads\Accident-Detection-System-main\Accident-Detection-System-main\model_weights (1).h5'
model = AccidentDetectionModel(model_json_file=model_json, model_weights_file=model_file)
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to send a WhatsApp message
def send_whatsapp_message(phone_number, message):
    try:
        # Schedule WhatsApp message 1 minute from the current time
        current_time = datetime.datetime.now()
        hours = current_time.hour
        minutes = current_time.minute + 1  # Add 1 minute to current time
        kit.sendwhatmsg(phone_number, message, hours, minutes)
        print(f"WhatsApp message scheduled for {phone_number}.")
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")

# Function to start the accident detection application
def startapplication():
    # Use video input from a file or camera (adjust as needed)
    video = cv2.VideoCapture(r'C:\Users\HP\Downloads\Accident-Detection-System-main\Accident-Detection-System-main\WhatsApp Video 2024-12-19 at 8.50.02 PM.mp4')  # For live camera, use: cv2.VideoCapture(0)
    last_message_time = 0  # Track the last time a message was sent
    cooldown_seconds = 60  # Cooldown period (in seconds) between messages


    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Preprocess the frame for prediction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # Make a prediction
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            # Display the prediction on the video frame
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

            # Send WhatsApp alert only if cooldown period has passed
            current_time = time.time()
            if current_time - last_message_time > cooldown_seconds:
                phone_number = "+918610749874"  # Replace with a valid WhatsApp number
                message = f"Accident detected with probability: {prob}%"
                send_whatsapp_message(phone_number, message)
                last_message_time = current_time  # Update the last message time

        # Display the video frame
        cv2.imshow('Video', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    startapplication()
