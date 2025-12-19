import cv2
from pose_module import PoseEstimator
from fall_rules import fall_rule_based
import time
import json
from db import get_active_cameras, insert_fall_alert
import multiprocessing as mp
import uuid
import base64
from PIL import Image 
from io import BytesIO
from collections import deque
import os

#load config.json
with open("/home/asadel/ASADEL PROJECTS/Fall_Detection/main/config.json", "r") as f:
    config = json.load(f)


# Paths
model_path = config["params"]["model_path"]
default_fps = config["params"]["default_fps"]

clip_before_seconds = config["alert"]["clip_before_seconds"] 
clip_after_seconds = config["alert"]["clip_after_seconds"]

CLIP_FOLDER ="/home/asadel/ASADEL PROJECTS/Fall_Detection_clip"
# create folder if it doesn't exist
os.makedirs(CLIP_FOLDER, exist_ok=True)

#to capture snapshot
def img_to_data_url(img):
    # Convert OpenCV (BGR) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)

    # Save to buffer in JPEG format
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")

    # Encode to base64
    img_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Format as Data URL
    return f"data:image/jpeg;base64,{img_encoded}"

def run_camera(camera_id, camera_name, rtsp_url, model_path, default_fps):
    print(f"\n Starting camera {camera_id}:{camera_name}")
    

    fall_memory = {}
    pose_estimator = PoseEstimator(model_path) 
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Failed to load Rtsp for {camera_id}:{camera_name}")
        return

    fps = 0
    prev_time = time.time()
    frame_counter = 0
   
    frame_buffer = deque()  # store frames for "before fall"
    recording = False
    record_start_time = None
    clip_frames = []
    before_frames = []

    alert_id = None
    snapshot_64 = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{camera_id}:{camera_name}] RTSP read failed")
                break

            now = time.time()

            raw_frame = frame.copy()

            frame_buffer.append((now, raw_frame))
            while frame_buffer and (now - frame_buffer[0][0] > clip_before_seconds):
                frame_buffer.popleft()

            # Update FPS
            frame_counter += 1
            if now - prev_time >= 1.0:
                fps = frame_counter / (now - prev_time)
                fps = int(fps)
                frame_counter = 0
                prev_time = now
                print(f"[{camera_id}:{camera_name}] FPS: {fps}")

            processed_frame, detections = pose_estimator.process_frame(frame)

            if recording:
                clip_frames.append(raw_frame)

                if now - record_start_time >= clip_after_seconds:
                    recording = False

                    # Combine before + after frames
                    all_frames = before_frames + clip_frames

                    if all_frames:
                        h, w, _ = all_frames[0].shape
                        clip_path = os.path.join(CLIP_FOLDER, f"{alert_id}.mp4")

                        out = cv2.VideoWriter(
                            clip_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps if fps > 0 else default_fps,
                            (w, h)
                        )

                        for f in all_frames:
                            out.write(f)
                        out.release()

                    else:
                        clip_path = None
                        print(f"[WARN] No frames for clip {alert_id}")

                    insert_fall_alert(
                        alert_id=alert_id,
                        camera_id=camera_id,
                        snapshot_64=snapshot_64,
                        clip_path=clip_path
                    )

                    clip_frames.clear()
                    before_frames.clear()

            # ---------------------------------
            # Fall detection
            # ---------------------------------
            for person_id, kpts, bbox in detections:
                status, alert_triggered = fall_rule_based(
                    person_id,
                    kpts,
                    bbox,
                    fps=fps if fps > 0 else default_fps,
                    fall_memory=fall_memory,
                    camera_id=camera_id,
                    camera_name=camera_name
                )

                if alert_triggered and not recording:
                    alert_id = str(uuid.uuid4())
                    snapshot_64 = img_to_data_url(frame)

                    before_frames = [f for _, f in frame_buffer]

                    recording = True
                    record_start_time = now
                    clip_frames = []

                # Draw status
                x1, y1, _, _ = map(int, bbox)
                color = (0, 255, 0) if status == "NORMAL" else \
                        (0, 255, 255) if status == "POTENTIAL_FALL" else \
                        (0, 0, 255)

                cv2.putText(
                    processed_frame, status,
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )

            # ---------------------------------
            # Display
            # ---------------------------------
            cv2.imshow(f"Camera {camera_id}", processed_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Camera {camera_id} stopped")


def main():
    cameras = get_active_cameras()

    if not cameras:
        print("No active cameras found in DB")
        return
    
    processes = []
    
    for cam in cameras:
        p = mp.Process(
            target= run_camera,
            args=(
                cam["CameraId"],
                cam['CameraName'],
                cam["RTSPUrl"],
                model_path,
                default_fps
            ),
            daemon=True
        )
        p.start()
        processes.append(p)
        print(f"Process started for {cam['CameraId']}")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[INFO] Terminating all camera process")
        for p in processes:
            p.terminate()

if __name__ =="__main__":
    mp.set_start_method("spawn", force = True)
    main()