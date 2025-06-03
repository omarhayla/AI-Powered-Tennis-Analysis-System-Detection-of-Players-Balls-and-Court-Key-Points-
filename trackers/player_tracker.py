from ultralytics import YOLO
import cv2
import pickle
import sys

sys.path.append('../')
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print("[INFO] YOLO model initialized with:", model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        print("[DEBUG] Filtering players based on court keypoints...")
        player_detections_first_frame = player_detections[0]
        print(f"[DEBUG] Detections in first frame: {player_detections_first_frame}")

        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        print(f"[DEBUG] Chosen players: {chosen_player}")

        filtered_player_detections = []
        for frame_idx, player_dict in enumerate(player_detections):
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if
                                    track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
            print(f"[DEBUG] Frame {frame_idx + 1} - Filtered detections: {filtered_player_dict}")
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        print("[DEBUG] Choosing players closest to court keypoints...")
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            print(f"[DEBUG] Player ID {track_id} center: {player_center}")

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                min_distance = min(min_distance, distance)
            distances.append((track_id, min_distance))
            print(f"[DEBUG] Player ID {track_id} - Closest distance: {min_distance}")

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        print(f"[DEBUG] Sorted distances: {distances}")

        # Choose the first 2 tracks
        if len(distances) < 2:
            print("[ERROR] Less than two players detected. Check input data or model.")
            raise ValueError("Could not select two distinct players from the first frame.")

        chosen_players = [distances[0][0], distances[1][0]]
        print(f"[DEBUG] Chosen players: {chosen_players}")
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            print("[INFO] Reading detections from stub file:", stub_path)
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        print("[INFO] Starting frame detection...")
        for i, frame in enumerate(frames):
            player_dict = self.detect_frame(frame)
            if not player_dict:
                print(f"[WARNING] No players detected in frame {i + 1}")
            else:
                print(f"[DEBUG] Frame {i + 1} detections: {player_dict}")
            player_detections.append(player_dict)

        if stub_path is not None:
            print("[INFO] Writing detections to stub file:", stub_path)
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        print("[DEBUG] Detecting players in frame...")
        results = self.model.track(frame, persist=True,conf=0.2)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            class_id = int(box.cls.tolist()[0])
            class_name = id_name_dict[class_id]

            # Debug: Check the class name
            print(f"[DEBUG] Detected object - Track ID: {track_id}, Class: {class_name}, BBox: {bbox}")

            if class_name == "Player1":  # Ensure this matches your trained model's label
                player_dict[track_id] = bbox

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        print("[INFO] Drawing bounding boxes on frames...")
        output_video_frames = []
        for frame_idx, (frame, player_dict) in enumerate(zip(video_frames, player_detections)):
            print(f"[DEBUG] Frame {frame_idx + 1} - Player detections: {player_dict}")

            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
