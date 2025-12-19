import numpy as np
from collections import deque
import json

# Load config.json
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

def debug_velocity(person_id, feats, vel_list, frame_count, fps, N, M, velocity_counter):
    """Debug function to print velocity calculation details"""
    print("\n" + "="*60)
    print(f"DEBUG for Person {person_id} - Frame {frame_count}")
    print("="*60)
    
    # 1. Check frame timing
    print(f"\n1. TIMING CHECK:")
    print(f"   Total frames stored: {len(feats)}")
    print(f"   Frame count: {frame_count}")
    print(f"   Velocity counter: {velocity_counter}/{N}")
    print(f"   Should compute when counter >= {N}")
    
    # 2. Check positions
    if len(feats) >= N+1:
        print(f"\n2. POSITION CHECK (N={N}):")
        old_y = feats[-N-1]["bbox_cy"]
        new_y = feats[-1]["bbox_cy"]
        print(f"   Position {N} frames ago (index -{N+1}): {old_y:.1f}")
        print(f"   Current position (index -1): {new_y:.1f}")
        print(f"   Change: {new_y - old_y:.1f} pixels")
        print(f"   Direction: {'DOWN' if new_y > old_y else 'UP'}")
        print(f"   Time interval: {N/fps:.3f} seconds")
        print(f"   Velocity if computed: {(new_y - old_y)/(N/fps):.1f} px/s")
    
    # 3. Check stored velocities
    print(f"\n3. VELOCITY LIST (M={M}):")
    if len(vel_list) > 0:
        for i, v in enumerate(vel_list):
            direction = "↓" if v < 0 else "↑" if v > 0 else "→"
            print(f"   v{i+1}: {v:7.1f} px/s {direction}")
        
        # 4. Check if increasing
        if len(vel_list) >= 2:
            print(f"\n4. INCREASING CHECK (need {M} velocities):")
            print(f"   Have {len(vel_list)} velocities, need {M}")
            
            if len(vel_list) >= M:
                recent_vels = vel_list[-M:]
                print(f"   Last {M} velocities: {[f'{v:.1f}' for v in recent_vels]}")
                
                # Check each pair
                for i in range(len(recent_vels)-1):
                    change = recent_vels[i+1] - recent_vels[i]
                    if recent_vels[i] != 0:
                        percent = (recent_vels[i+1] / recent_vels[i] - 1) * 100
                    else:
                        percent = 0
                    trend = "↑" if change > 0 else "↓" if change < 0 else "="
                    print(f"   v{i+2} - v{i+1}: {change:7.1f} ({percent:6.1f}%) {trend}")
                
                # Your current check
                strictly_increasing = recent_vels == sorted(recent_vels) and recent_vels[0] < recent_vels[-1]
                print(f"\n   Strict sorted check: {strictly_increasing}")
                
                # Better check
                increasing_count = sum(1 for i in range(len(recent_vels)-1) 
                                      if recent_vels[i+1] > recent_vels[i])
                print(f"   Increasing pairs: {increasing_count}/{len(recent_vels)-1}")
                better_check = increasing_count >= (len(recent_vels) - 1)
                print(f"   Better check (all pairs increasing): {better_check}")
    else:
        print("   No velocities stored yet")
    
    print("="*60 + "\n")

def get_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))

def compute_features(kpts, box):
    nose = kpts[0]
    left_shoulder, right_shoulder = kpts[5], kpts[6]
    left_hip, right_hip = kpts[11], kpts[12]
    left_ankle, right_ankle = kpts[15], kpts[16]

    shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
    hip_mid = np.mean([left_hip, right_hip], axis=0)
    ankle_mid = np.mean([left_ankle, right_ankle], axis=0)

    torso_angle = get_angle(shoulder_mid, hip_mid)
    body_height = np.linalg.norm(nose - ankle_mid)
    head_vs_ankle = nose[1] - ankle_mid[1]

    x1, y1, x2, y2 = box
    bbox_h = y2 - y1
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
            "vel": deque(maxlen=M),
            "velocity_counter": 0,
            "frame_count": 0  # ADDED: Total frame counter for debugging
        }

    # Increment frame count
    fall_memory[person_id]["frame_count"] += 1
    frame_count = fall_memory[person_id]["frame_count"]
    
    # Save latest frame features
    fall_memory[person_id]["frames"].append(feat)
    feats = list(fall_memory[person_id]["frames"])

    if len(feats) < N+1:
        return "NORMAL"
    
    # DEBUG: Run for first person only, every N frames
    if person_id == 0 and frame_count % N == 0:  # Print every N frames
        vel_list = list(fall_memory[person_id]["vel"])
        debug_velocity(person_id, feats, vel_list, frame_count, fps, N, M, 
                      fall_memory[person_id]["velocity_counter"])
    
    # ------------------------------------------------------------
    # PART 1 — VELOCITY CALCULATION
    # ------------------------------------------------------------
    fall_memory[person_id]["velocity_counter"] += 1

    if fall_memory[person_id]["velocity_counter"] >= N:
        fall_memory[person_id]["velocity_counter"] = 0

        if len(feats) >= N:
            old_y = feats[-N-1]["bbox_cy"]   # N frames ago
            new_y = feats[-1]["bbox_cy"]     # current frame
            velocity = (new_y - old_y) / (N / fps)
            fall_memory[person_id]["vel"].append(velocity)

    vel_list = list(fall_memory[person_id]["vel"])
    vel_increasing = False
    
    if len(vel_list) >= M:
        recent_vels = vel_list[-M:]
        # Your original check (strict)
        vel_increasing = recent_vels == sorted(recent_vels) and recent_vels[0] < recent_vels[-1]
    
    # --------------------------------------------------------
    # PART 2 — BBOX COLLAPSE
    # --------------------------------------------------------
    curr_feat = feats[-1]
    prev_feat = feats[-N-1]

    prev_h = prev_feat["bbox_h"]
    curr_h = curr_feat["bbox_h"]
    prev_w = prev_feat["bbox_w"]
    curr_w = curr_feat["bbox_w"]

    height_halved = curr_h < height_drop_ratio * prev_h
    width_increased = curr_w > width_increase_ratio * prev_w
    height_dropping = height_halved or width_increased
    
    # --------------------------------------------------------
    # PART 3 — KEYPOINT LOGIC
    # --------------------------------------------------------
    heights = [f["body_height"] for f in feats]
    angles = [f["torso_angle"] for f in feats]
    
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
    
    # DEBUG: Print decision
    if person_id == 0 and frame_count % N == 0:
        print(f"\nDECISION CHECK for Person {person_id}:")
        print(f"  vel_increasing: {vel_increasing}")
        print(f"  height_dropping: {height_dropping}")
        print(f"  rapid_height_loss: {rapid_height_loss}")
        print(f"  head_inversion: {head_inversion}")
        print(f"  bent_posture: {bent_posture}")
        print(f"  keypoint_fall: {keypoint_fall}")
        print(f"  FINAL: vel_increasing or height_dropping or rapid_height_loss = "
              f"{vel_increasing or height_dropping or rapid_height_loss}")
        print("-" * 40)
    
    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------
    # NOTE: Your logic uses OR instead of AND - is this intentional?
    if (vel_increasing or height_dropping or rapid_height_loss):
        return "FALL"

    return "NORMAL"