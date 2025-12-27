import pickle
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # =======================================================
    #   SAFE INTERPOLATION - NEVER CRASHES
    # =======================================================
    def interpolate_ball_positions(self, ball_positions):

        cleaned = []

        for frame in ball_positions:

            # Case 1: valid dict containing ball
            if isinstance(frame, dict):
                if 1 in frame and isinstance(frame[1], list) and len(frame[1]) == 4:
                    cleaned.append(frame[1])
                    continue
                if "ball" in frame and isinstance(frame["ball"], list) and len(frame["ball"]) == 4:
                    cleaned.append(frame["ball"])
                    continue

            # Case 2: list formatted correctly
            if isinstance(frame, list):
                if len(frame) == 4:
                    cleaned.append(frame)
                    continue
                if len(frame) == 1 and isinstance(frame[0], list) and len(frame[0]) == 4:
                    cleaned.append(frame[0])
                    continue

            # Fall-back for any bad case
            cleaned.append([np.nan, np.nan, np.nan, np.nan])

        # ---- Convert to DataFrame ----
        df = pd.DataFrame(cleaned, columns=['x1', 'y1', 'x2', 'y2']).astype(float)

        # If everything is NaN: no ball found at all
        if df.isnull().all().all():
            print("⚠️ No ball positions found — interpolation skipped.")
            return [{1: None} for _ in cleaned]

        # ---- Interpolate missing values ----
        df = df.interpolate().bfill()

        # ---- Convert back ----
        final_positions = [{1: row.tolist()} for _, row in df.iterrows()]
        return final_positions

    # =======================================================
    #   BALL SHOT FRAME DETECTION
    # =======================================================
    def get_ball_shot_frames(self, ball_positions):

        cleaned = []

        for frame in ball_positions:

            if isinstance(frame, dict) and 1 in frame and isinstance(frame[1], list) and len(frame[1]) == 4:
                cleaned.append(frame[1])
                continue

            cleaned.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(cleaned, columns=['x1', 'y1', 'x2', 'y2']).astype(float)

        # If no detection → skip hit detection
        if df.isnull().all().all():
            print("⚠️ No valid ball boxes — hit detection skipped.")
            return []

        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_avg'] = df['mid_y'].rolling(window=5, min_periods=1).mean()
        df['delta_y'] = df['mid_y_avg'].diff()

        hit_frames = []
        threshold = 25

        for i in range(1, len(df) - int(threshold * 1.2)):

            upward = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i + 1] > 0
            downward = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i + 1] < 0

            if upward or downward:

                count = 0
                for j in range(i+1, i + int(threshold * 1.2) + 1):
                    if upward and df['delta_y'].iloc[j] > 0:
                        count += 1
                    if downward and df['delta_y'].iloc[j] < 0:
                        count += 1

                if count > threshold - 1:
                    hit_frames.append(i)

        return hit_frames

    # =======================================================
    #   STUB OR YOLO DETECTION
    # =======================================================
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                print("👉 Loaded ball stubs!")
                return pickle.load(f)

        ball_detections = []

        for frame in frames:
            ball_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)
            print("👉 Saved ball stubs!")

        return ball_detections

    # =======================================================
    #   YOLO SINGLE FRAME
    # =======================================================
    def detect_frame(self, frame):

        results = self.model(frame, conf=0.15)[0]
        ball_dict = {}

        for box in results.boxes:
            # take first detection
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            ball_dict[1] = [x1, y1, x2, y2]
            break

        return ball_dict

    # =======================================================
    #   DRAW BOUNDING BOXES
    # =======================================================
    def draw_bboxes(self, video_frames, detections):

        out_frames = []

        for frame, ball_dict in zip(video_frames, detections):

            if 1 in ball_dict and ball_dict[1] is not None:

                x1, y1, x2, y2 = ball_dict[1]

                cv2.putText(
                    frame,
                    "Ball",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2
                )

            out_frames.append(frame)

        return out_frames

