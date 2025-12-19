# Fall Detection System

A real-time **Fall Detection System** built using computer vision and pose estimation. The system processes multiple RTSP camera streams in parallel, detects human falls using rule-based logic on pose keypoints, records evidence clips, and stores alerts in a MySQL database.

---

## 🚀 Features

* ✅ Real-time fall detection using pose keypoints
* 🎥 Multi-camera RTSP stream support (parallel processing)
* 🧠 Rule-based fall confirmation (multi-frame validation)
* ⏱ Time-based video clipping (e.g. 2s before + 3s after fall)
* 🖼 Snapshot capture at fall moment (Base64)
* 💾 Alert storage in MySQL database
* 🧾 Detailed debug logs for detection states
* ⚙️ Config-driven system (JSON)

---

## 🏗 System Architecture

```
RTSP Camera(s)
      ↓
OpenCV Video Capture
      ↓
Pose Estimation (PoseEstimator)
      ↓
Fall Rule Engine (fall_rule_based)
      ↓
Alert Trigger
   ↙         ↘
Snapshot     Video Clip (raw)
      ↓
MySQL Alerts Table
```

Each camera runs in its **own process** using Python multiprocessing.

---

## 📂 Project Structure

```
Fall_Detection/
│
├── main/
│   ├── main.py              # Main entry point
│   ├── db.py                # Database operations
│   ├── fall_rules.py        # Fall detection logic
│   ├── pose_module.py       # Pose estimation wrapper
│   ├── config.json          # System configuration
│
├── Fall_Detection_clip/     # Saved video clips
└── README.md
```

---

⚙️ Configuration (config.json)

The system is fully config-driven. All database credentials, model paths, alert timings, and fall detection thresholds are controlled via config.json.

1️⃣ Parameters Section
2️⃣ Alert Configuration
3️⃣ Fall Detection Thresholds
4️⃣ Database Configuration
---

## 🧠 Fall Detection Logic

Fall detection is **not single-frame based**. A fall is confirmed only when:

* Rapid downward motion is detected
* Head inversion or posture abnormality occurs
* Conditions persist across multiple consecutive frames

### States:

* **NORMAL** – Person is stable
* **POTENTIAL_FALL** – Early indicators detected
* **FALLING** – Confirmed fall (alert triggered)

Debug logs clearly indicate transitions between these states.

---

## 🎞 Video Clip Handling

* Frames are buffered using a **time-based deque**
* On fall confirmation:

  * Last `N` seconds are taken from buffer (before fall)
  * Next `M` seconds are recorded live (after fall)
* Resulting clip is **raw video only** (no bounding boxes or keypoints)

This avoids blocking the live feed and ensures smooth playback.

---

## ▶️ How to Run

1. Create virtual environment
```bash
python3 -m venv venv
```

2. Activate virtual environment
```bash
source venv/bin/activate
```

3. Start the system
```bash
python main/main.py
```

Each active camera from the database will start in its own process.

---

## 🧪 Debugging & Logs

* FPS printed per camera
* State transitions logged:

  * POTENTIAL_FALL
  * FALL CONFIRMED
* Frame buffer size visibility

---

## 🛑 Graceful Shutdown

* Press `ESC` to stop individual camera windows
* `Ctrl + C` stops all camera processes safely

---

