import cv2
from pose_module import PoseEstimator
from fall_rules import fall_rule_based, fall_memory
import time
import json

#load config.json
with open("/home/asadel/ASADEL PROJECTS/Fall_Detection/main/config.json", "r") as f:
    config = json.load(f)


# Paths
model_path = config["params"]["model_path"]
video_path = config["params"]["video_path"]
default_fps = config["params"]["default_fps"]

pose_estimator = PoseEstimator(model_path)
cap = cv2.VideoCapture(video_path)

# -----------------------
# VIDEO SPEED CONTROLLER
# -----------------------
#speed = 2.0  # <--- change here: 0.2 super slow, 1.0 normal, 2.0 fast

#video_fps = cap.get(cv2.CAP_PROP_FPS)
#if video_fps <= 0:
#    video_fps = 30

#delay_ms = int((1000 / video_fps) / speed)
#delay_ms = max(1, delay_ms)
# -----------------------

fps = 0
prev_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, detections = pose_estimator.process_frame(frame)

    # Update FPS
    frame_counter += 1
    current_time = time.time()
    if current_time - prev_time >= 1.0:  # update every second
        fps = frame_counter / (current_time - prev_time)
        fps = int(fps)  # optional: round to integer
        frame_counter = 0
        prev_time = current_time
        print("FPS:", fps)

    # Apply fall detection
    for person_id, kpts, bbox in detections:
        status = fall_rule_based(person_id, kpts, bbox, fps=fps if fps > 0 else default_fps)
        print(f"Person {person_id} → frames in memory:", len(fall_memory[person_id]["frames"]))
        x1, y1, _, _ = map(int, bbox)
        # Correct color mapping
        if status == "NORMAL":
            color = (0, 255, 0)          # Green
        elif status == "POTENTIAL_FALL":
            color = (0, 255, 255)        # Yellow
        elif status == "FALLING":
            color = (0, 0, 255)          # Red
            
        else:
            color = (255, 255, 255)   
        cv2.putText(frame, f"{status}", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
