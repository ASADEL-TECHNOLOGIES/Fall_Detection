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

# NEW: Fall velocity thresholds
MIN_FALL_SPEED = 50  # pixels/second - minimum downward speed
MIN_FALL_ACCELERATION = 20  # pixels/second² - minimum acceleration
CONFIRMATION_FRAMES = 5  # Number of consecutive frames to confirm fall

fall_memory = {}   # memory per person

def debug_velocity(person_id, feats, vel_list, frame_count, fps, N, M):
    """Simplified debug function"""
    print("\n" + "="*60)
    print(f"DEBUG for Person {person_id} - Frame {frame_count}")
    print("="*60)
    
    # 1. Frame timing
    print(f"\n1. TIMING:")
    print(f"   Total frames: {frame_count}")
    print(f"   Frames stored: {len(feats)}")
    print(f"   Should compute at: {N}, {2*N}, {3*N}...")
    print(f"   frame_count % N = {frame_count % N}")
    
    
    # 3. Velocity list
    print(f"\n3. STORED VELOCITIES (M={M}):")
    if len(vel_list) > 0:
        for i, v in enumerate(vel_list):
            direction = "↓" if v > 0 else "↑" if v < 0 else "→"
            print(f"   v{i+1}: {v:7.1f} px/s {direction}")
        
        if len(vel_list) >= M:
            recent_vels = vel_list[-M:]
            print(f"\n4. FALL VELOCITY CHECK (last {M}):")
            print(f"   Velocities: {[f'{v:.1f}' for v in recent_vels]}")
            
            avg_vel = np.mean(recent_vels)
            max_vel = max(recent_vels)
            acceleration = recent_vels[-1] - recent_vels[0] if len(recent_vels) > 1 else 0
            
            print(f"   Average velocity: {avg_vel:.1f} px/s")
            print(f"   Max velocity: {max_vel:.1f} px/s")
            print(f"   Acceleration: {acceleration:.1f} px/s²")
            print(f"   Rapid downward: {avg_vel > MIN_FALL_SPEED and max_vel > MIN_FALL_SPEED}")
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
    head_vs_ankle = nose[1] - ankle_mid[1]  # Negative when standing, positive when head below feet

    x1, y1, x2, y2 = box
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    bbox_cy = (y1 + y2) / 2

    return {
        "torso_angle": torso_angle,
        "body_height": body_height,
        "head_vs_ankle": head_vs_ankle,
        "nose_y": nose[1],
        "ankle_y": ankle_mid[1],
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "bbox_cy": bbox_cy
    }

def check_rapid_downward_movement(velocities, min_speed=MIN_FALL_SPEED, min_accel=MIN_FALL_ACCELERATION):
    """
    Check if velocities indicate falling (rapid downward movement with acceleration)
    
    Args:
        velocities: List of downward velocities (positive = downward in image coords)
        min_speed: Minimum average speed to consider fall (px/s)
        min_accel: Minimum acceleration to consider fall (px/s²)
    
    Returns:
        True if movement indicates a fall
    """
    if len(velocities) < 2:
        return False
    
    # Check recent velocities
    recent_vels = velocities[-M:] if len(velocities) >= M else velocities
    
    # All velocities should be positive (downward movement)
    if not all(v > 0 for v in recent_vels):
        return False
    
    # Calculate metrics
    avg_velocity = np.mean(recent_vels)
    max_velocity = max(recent_vels)
    
    # Check if accelerating (velocity increasing over time)
    acceleration = recent_vels[-1] - recent_vels[0]
    
    # Fall criteria: fast downward movement
    is_fast_enough = avg_velocity > min_speed and max_velocity > min_speed
    is_accelerating = acceleration > min_accel or max_velocity > min_speed * 1.5
    
    return is_fast_enough and is_accelerating

memory_frames = N * M

def fall_rule_based(person_id, kpts, box, fps):
    global fall_memory

    feat = compute_features(kpts, box)

    if person_id not in fall_memory:
        fall_memory[person_id] = {
            "frames": deque(maxlen=memory_frames),
            "vel": deque(maxlen=M),
            "frame_count": 0,
            "fall_frames": 0,  # Count consecutive fall detections
            "alert_sent": False  # Track if alert already sent
        }

    fall_memory[person_id]["frame_count"] += 1
    frame_count = fall_memory[person_id]["frame_count"]
    
    fall_memory[person_id]["frames"].append(feat)
    feats = list(fall_memory[person_id]["frames"])

    if len(feats) < N+1:
        return "NORMAL"
    
    # ------------------------------------------------------------
    # 1. VELOCITY COMPUTATION (every N frames)
    # ------------------------------------------------------------
    if frame_count % N == 0 and len(feats) >= N+1:
        old_y = feats[-N-1]["bbox_cy"]
        new_y = feats[-1]["bbox_cy"]
        # Positive velocity = downward movement (Y increases downward in image)
        velocity = (new_y - old_y) / (N / fps)
        
        fall_memory[person_id]["vel"].append(velocity)
        
        # DEBUG: Print when velocity is computed
        if person_id == 0:
            print(f"\n[VELOCITY COMPUTED AT FRAME {frame_count}]")
            print(f"  Old Y: {old_y:.1f}, New Y: {new_y:.1f}")
            print(f"  Change: {new_y - old_y:.1f} pixels")
            print(f"  Velocity: {velocity:.1f} px/s ({'DOWNWARD' if velocity > 0 else 'UPWARD'})")
    
    # ------------------------------------------------------------
    # 2. DEBUG PRINT (with updated velocities)
    # ------------------------------------------------------------
    if frame_count % N == 0:
        vel_list = list(fall_memory[person_id]["vel"])
        debug_velocity(person_id, feats, vel_list, frame_count, fps, N, M)
    
    # ------------------------------------------------------------
    # 3. CHECK VELOCITY FOR RAPID DOWNWARD MOVEMENT
    # ------------------------------------------------------------
    vel_list = list(fall_memory[person_id]["vel"])
    rapid_downward = False
    
    if len(vel_list) >= M:
        rapid_downward = check_rapid_downward_movement(vel_list)
        
    # ------------------------------------------------------------
    # 4. BBOX COLLAPSE CHECK
    # ------------------------------------------------------------
    curr_feat = feats[-1]
    prev_feat = feats[-N-1]

    prev_h = prev_feat["bbox_h"]
    curr_h = curr_feat["bbox_h"]
    prev_w = prev_feat["bbox_w"]
    curr_w = curr_feat["bbox_w"]

    height_halved = curr_h < height_drop_ratio * prev_h
    width_increased = curr_w > width_increase_ratio * prev_w
    height_dropping = height_halved or width_increased
    
    # ------------------------------------------------------------
    # 5. KEYPOINT-BASED CHECKS
    # ------------------------------------------------------------
    heights = [f["body_height"] for f in feats]
    angles = [f["torso_angle"] for f in feats]
    
    rapid_height_loss = False
    bent_posture = False
    head_inversion = False
    
    # Check if body height dropped significantly
    if len(heights) >= 5:
        height_drop = (heights[-5] - heights[-1]) / max(1, heights[-5])
        rapid_height_loss = height_drop > height_drop_kp
    
    # Check if person is bent/horizontal
    if len(angles) >= 5:
        avg_angle = np.mean(angles[-5:])
        bent_posture = avg_angle > bent_angle
    
    # Check if head is below feet (inverted posture)
    # head_vs_ankle is negative when standing, positive when head below ankles
    head_inversion = curr_feat["head_vs_ankle"] > 0
    
    keypoint_fall = rapid_height_loss or head_inversion or bent_posture
    
    # ------------------------------------------------------------
    # 6. FALL DECISION - MULTI-CRITERIA WITH CONFIRMATION
    # ------------------------------------------------------------
    
    # Count fall indicators
    velocity_indicator = rapid_downward
    other_indicators = []
    
    if height_dropping:
        other_indicators.append("height_dropping")

    if rapid_height_loss:
        other_indicators.append("rapid_height_loss")

    if head_inversion:
        other_indicators.append("head_inversion")

    if bent_posture:
        other_indicators.append("bent_posture")

    # 2. Apply new rule: velocity is mandatory
    fall_indicators = 0
    indicators_detail = []

    # velocity must be present
    if velocity_indicator:
        fall_indicators += 1
        indicators_detail.append("rapid_downward")

    # at least one more indicator from the others
    if len(other_indicators) >= 1:
        fall_indicators += 1
        indicators_detail.extend(other_indicators)
    
    
    # CORRECTED LOGIC: Require at least 2 indicators for potential fall
    if fall_indicators >= 2:
        fall_memory[person_id]["fall_frames"] += 1
        
        # Confirm fall only after consecutive detections
        if fall_memory[person_id]["fall_frames"] >= CONFIRMATION_FRAMES:
            if not fall_memory[person_id]["alert_sent"]:
                fall_memory[person_id]["alert_sent"] = True
                print(f"\n{'='*60}")
                print(f"  FALL CONFIRMED for Person {person_id}!")
                print(f"   Indicators: {indicators_detail}")
                print(f"   Confirmed over {fall_memory[person_id]['fall_frames']} frames")
                print(f"{'='*60}\n")
            return "FALLING"
        else:
            return "POTENTIAL_FALL"
    else:
        # Reset fall counter if conditions not met
        fall_memory[person_id]["fall_frames"] = 0
        fall_memory[person_id]["alert_sent"] = False
        return "NORMAL"