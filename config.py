
BOT_TOKEN = "8638083992:AAGvtNwcaK9Deqr-j084EvCNaSx7E3y6ioo"       
CHAT_ID   = "2057919929"  

# --- Camera / Video Source ---
# For live webcam          : VIDEO_SOURCE = 0
# For a video file         : VIDEO_SOURCE = r"C:\path\to\video.mp4"
# For CCTV via RTSP        : VIDEO_SOURCE = "rtsp://admin:pass@192.168.1.64:554/stream"
VIDEO_SOURCE = r"D:\SUDHARSAN\PROJECTS\accident_detection_system\accident clip.mp4"

CAMERA_NAME = "Cam 1"

# --- Model Files ---
MODEL_JSON    = "model.json"
MODEL_WEIGHTS = "model_weights.weights.h5"  

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.50    # Only alert above this confidence (0.0 - 1.0)
COOLDOWN_SECONDS     = 60      # Minimum seconds between alerts
FRAME_SKIP           = 5       # Process every Nth frame (higher = faster)

# --- Alert Tiers ---
# Above HIGH_CONFIDENCE        → text message + photo
# Above CONFIDENCE_THRESHOLD   → text message only
# Below CONFIDENCE_THRESHOLD   → no alert
HIGH_CONFIDENCE = 0.90