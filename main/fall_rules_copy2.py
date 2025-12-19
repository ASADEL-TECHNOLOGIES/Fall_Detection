import numpy as np
from collections import deque
import json

#load config.json
with open("/home/asadel/ASADEL PROJECTS/Fall_Detection/main/config.json", "r") as f:
    config = json.load(f)

# Paths
N = config["params"]["frames_per_velocity"]  
M = config["params"]["velocity_history_length"]


height_drop_ratio = config["thresholds"]["bbox"]["height_drop_ratio"]
width_increase_ratio = config["thresholds"]["bbox"]["width_increase_ratio"]

bent_angle = config["thresholds"]["keypoint"]["bent_posture_angle"]
height_drop_kp = config["thresholds"]["keypoint"]["height_drop"]
head_inversion_threshold = config["thresholds"]["posture"]["head_inversion_threshold"]


fall_memory = {}   # memory per person

def get_angle(p1, p2):  #(x,y) coordinates 
    #p1 =(x1, y1)
    #p2 =(x2, y2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1] # horizontal distance , vertical distance
    return np.degrees(np.arctan2(abs(dx), abs(dy))) #return the angle between points

def compute_features(kpts, box):
    nose = kpts[0]
    left_shoulder, right_shoulder = kpts[5], kpts[6]
    left_hip, right_hip = kpts[11], kpts[12]
    left_ankle, right_ankle = kpts[15], kpts[16]

    shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
    hip_mid = np.mean([left_hip, right_hip], axis=0)
    ankle_mid = np.mean([left_ankle, right_ankle], axis=0)

    torso_angle = get_angle(shoulder_mid, hip_mid)
    body_height = np.linalg.norm(nose - ankle_mid) # np.linalg.norm used to calculate size or length
    head_vs_ankle = nose[1] - ankle_mid[1]

    x1, y1, x2, y2 = box
    bbox_h  = y2 - y1
    bbox_w = x2 - x1
    bbox_cy = (y1 + y2) / 2

    return {
        "torso_angle": torso_angle,
        "body_height": body_height,
        "head_vs_ankle": head_vs_ankle,
        "nose_y": nose[1],
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "bbox_cy": bbox_cy
    }

memory_frames = N * M
def fall_rule_based(person_id, kpts, box, fps):
    global fall_memory

    feat = compute_features(kpts, box)

    # ---------- INIT MEMORY PER PERSON ---------- #
    if person_id not in fall_memory:
        fall_memory[person_id] = {
            "frames": deque(maxlen=memory_frames),
            "vel": deque(maxlen=M),     # store last 5 velocities
            "velocity_counter": 0      # NEW: frame counter for velocity calc

        }

    # Save latest frame features
    fall_memory[person_id]["frames"].append(feat)
    feats = list(fall_memory[person_id]["frames"])

    if len(feats) < N+1:
        return "NORMAL"
    
    # ------------------------------------------------------------
    # PART 1 — ORIGINAL 3-LOGIC (velocity, Y increase, height drop)
    # --------------------------------------------------------

    # ------CONDITION 1: Velocity per frames

    fall_memory[person_id]["velocity_counter"] += 1

    if fall_memory[person_id]["velocity_counter"] >= N:
        fall_memory[person_id]["velocity_counter"] = 0

        if len(feats) >= N:
            old_y = feats[-N-1]["bbox_cy"]   # N frames ago
            new_y = feats[-1]["bbox_cy"]   # current frame

            velocity = (new_y - old_y) / (N / fps)

            fall_memory[person_id]["vel"].append(velocity)

    vel_list = list(fall_memory[person_id]["vel"])

    vel_increasing = False
    if len(vel_list) >= M:
        recent_vels = vel_list[-M:]
        vel_increasing = recent_vels == sorted(recent_vels) and recent_vels[0] < recent_vels[-1]
    
    
    # ----- CONDITION 2: Horizontal Collapse (much more accurate) -----
    # Bounding box values
    # Get current frame and frame N frames ago
    curr_feat = feats[-1]      # Current frame
    prev_feat = feats[-N-1]       # N frames ago (same as velocity comparison)


    prev_h = prev_feat["bbox_h"]
    curr_h = curr_feat["bbox_h"]

    # We must compute bbox width also (x2 - x1)
    prev_w = prev_feat["bbox_w"]
    curr_w = curr_feat["bbox_w"]

    # Condition A: Height collapses to half
    height_halved = curr_h < height_drop_ratio * prev_h  #20% percent decrement of bbox 

    # Condition B: Width increases (person becomes horizontal)
    width_increased = curr_w > width_increase_ratio * prev_w   # 20% wider

    # Final bbox fall condition
    height_dropping = height_halved # (or/and width_increased  )


    # --------------------------------------------------------
    # PART 2 — KEYPOINT LOGIC (angle, height drop, posture)
    # --------------------------------------------------------
    heights = [f["body_height"] for f in feats]
    angles  = [f["torso_angle"] for f in feats]

    # ADD SAFETY CHECKS:
    rapid_height_loss = False
    bent_posture = False
    head_inversion = feats[-1]["head_vs_ankle"] > head_inversion_threshold
    
    if len(heights) >= 5:
        height_drop = (heights[-5] - heights[-1]) / max(1, heights[-5])
        rapid_height_loss = height_drop > height_drop_kp
    
    if len(angles) >= 5:
        avg_angle = np.mean(angles[-5:])
        bent_posture = avg_angle > bent_angle
    
    keypoint_fall = rapid_height_loss or head_inversion or bent_posture

    # --------------------------------------------------------
    # FINAL DECISION (COMBINED SYSTEM)
    # --------------------------------------------------------
    if (
        (vel_increasing or height_dropping)          # your original logic
        or (rapid_height_loss)     # keypoint logic
    ):
        return "FALL"

    return "NORMAL"
