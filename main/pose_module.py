import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import json


with open("/home/asadel/ASADEL PROJECTS/Fall_Detection/main/config.json", "r") as f:
    config = json.load(f)

conf = config["params"]["person_conf_threshold"]
keypoint_conf = config["params"]["keypoint_conf_threshold"]
min_visible_kpts = config["params"]["min_visible_keypoints"]


class PoseEstimator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kalman_filters = {}

        # skeleton connections (COCO format)
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (11, 12), (5, 11), (6, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    def create_kf(self, x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0, 0], dtype=float)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        kf.P *= 10
        kf.R *= 0.1
        kf.Q *= 0.01
        return kf

    def process_frame(self, frame):
        """Run pose estimation with ByteTrack and return results for each person"""

        results = self.model.track(
            frame,
            persist=True,
            tracker='bytetrack.yaml',
            conf=conf,
            classes=[0]
        )

        output = []

        for r in results:
            boxes = r.boxes
            kpts_xy = r.keypoints.xy
            kpts_conf = r.keypoints.conf

            if boxes.id is None:
                continue

            for box, kpts, confs in zip(boxes, kpts_xy, kpts_conf):
                bbox = box.xyxy[0].cpu().numpy()
                kpts_array = kpts.cpu().numpy()
                conf_array = confs.cpu().numpy()

                person_id = int(box.id.cpu().numpy()[0])

                # -------------------------------
                # Filter by confidence
                # -------------------------------
                visible_mask = conf_array > keypoint_conf
                visible_count = np.sum(visible_mask)

                if visible_count < min_visible_kpts:
                    continue  # skip unreliable pose

                # -------------------------------
                # Init Kalman (once per ID)
                # -------------------------------
                if person_id not in self.kalman_filters:
                    self.kalman_filters[person_id] = [
                        self.create_kf(x, y) for x, y in kpts_array
                    ]

                # -------------------------------
                # Kalman update (NO RESET)
                # -------------------------------
                smooth_kpts = []

                for i, (kf, (x, y)) in enumerate(
                    zip(self.kalman_filters[person_id], kpts_array)
                ):
                    kf.predict()

                    if visible_mask[i]:
                        kf.update([x, y])

                    smooth_kpts.append((kf.x[0], kf.x[1]))

                # -------------------------------
                # Drawing
                # -------------------------------
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                for i, (sx, sy) in enumerate(smooth_kpts):
                    if visible_mask[i]:   # draw ONLY high-confidence keypoints
                        cv2.circle(frame, (int(sx), int(sy)), 3, (0, 255, 0), -1)


                for i, j in self.skeleton:
                    if visible_mask[i] and visible_mask[j]:
                        x1_s, y1_s = smooth_kpts[i]
                        x2_s, y2_s = smooth_kpts[j]
                        cv2.line(frame,
                                (int(x1_s), int(y1_s)),
                                (int(x2_s), int(y2_s)),
                                (0, 0, 255), 2)

                output.append((person_id, smooth_kpts, bbox))

        return frame, output
