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
            "vel": deque(maxlen=M)     # store last 5 velocities
        }

    # Save latest frame features
    fall_memory[person_id]["frames"].append(feat)
    feats = list(fall_memory[person_id]["frames"])

    if len(feats) < 10:
        return "NORMAL"
    
    # ------------------------------------------------------------
    # PART 1 — ORIGINAL 3-LOGIC (velocity, Y increase, height drop)
    # ------------------------------------------------------------

    prev_feats = feats[:N]
    curr_feats = feats[N:2*N]

    prev_y = [f["bbox_cy"] for f in prev_feats]
    curr_y = [f["bbox_cy"] for f in curr_feats]

    # CONDITION 1: Compute velocity
    prev_vel = (prev_y[-1] - prev_y[0]) / (N / fps)  # compute velocity using first and last frame 
    curr_vel = (curr_y[-1] - curr_y[0]) / (N / fps)

    fall_memory[person_id]["vel"].append(curr_vel)   # store n velocities in a list 
    vel_list = list(fall_memory[person_id]["vel"])

    vel_increasing = False               #check if velocity is consecutively increasing or not
    if len(vel_list) == M:
        vel_increasing = vel_list == sorted(vel_list) and vel_list[0] < vel_list[-1]

    #CONDITION 2: Y Increasing
    y_increasing = (curr_y[-1] - curr_y[0]) > 0

    # ----- CONDITION 3: Horizontal Collapse (much more accurate) -----
    # Bounding box values
    prev_h = np.mean([f["bbox_h"] for f in prev_feats])
    curr_h = np.mean([f["bbox_h"] for f in curr_feats])

    # We must compute bbox width also (x2 - x1)
    prev_w = np.mean([(f["bbox_w"]) for f in prev_feats])
    curr_w = np.mean([(f["bbox_w"]) for f in curr_feats])

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

    #Calculating height drop of a person using its key point
    height_drop = (heights[-5] - heights[-1]) / max(1, heights[-5])

    #Storing last 5 frame torso angle 
    avg_angle = np.mean(angles[-5:])

    head_inversion = feats[-1]["head_vs_ankle"] > head_inversion_threshold
    bent_posture = avg_angle > bent_angle
    rapid_height_loss = height_drop > height_drop_kp

    # --------------------------------------------------------
    # FINAL DECISION (COMBINED SYSTEM)
    # --------------------------------------------------------
    if (
        (vel_increasing and y_increasing or height_dropping )          # your original logic
        or
        (rapid_height_loss or head_inversion or bent_posture)          # keypoint logic
    ):
        return "FALL"

    return "NORMAL"
