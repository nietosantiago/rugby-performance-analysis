import numpy as np
import cv2
from scipy.spatial import distance
from sklearn.cluster import KMeans
from collections import OrderedDict
import config

class PlayerTracker:
    def __init__(self, max_disappeared=15, max_distance=100):
        self.next_object_id = 1
        self.objects = OrderedDict()  # id: centroid
        self.disappeared = OrderedDict()
        self.bboxes = OrderedDict() # id: bbox
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Color tracking for team assignment
        self.team_model_trained = False
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.player_colors = [] # Store colors to train kmeans
        self.player_teams = {} # id: team label

    def extract_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # Take upper body center crop to find jersey color
        h, w = y2 - y1, x2 - x1
        crop_y1 = max(0, y1 + int(h * 0.1))
        crop_y2 = min(frame.shape[0], y1 + int(h * 0.5))
        crop_x1 = max(0, x1 + int(w * 0.3))
        crop_x2 = min(frame.shape[1], x1 + int(w * 0.7))
        
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return np.array([0, 0, 0])
            
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Average color in the crop
        avg_color = np.mean(crop_rgb, axis=(0, 1))
        return avg_color

    def register(self, centroid, bbox, color):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.bboxes[object_id] = bbox
        self.next_object_id += 1
        
        # We sample up to 100 colors to train team clustering
        if not self.team_model_trained and len(self.player_colors) < 100:
            self.player_colors.append(color)
            if len(self.player_colors) >= 50:
                self.kmeans.fit(self.player_colors)
                self.team_model_trained = True

        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]

    def update(self, frame, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.bboxes, self.player_teams

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes = []
        input_colors = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
            input_bboxes.append(det["bbox"])
            input_colors.append(self.extract_color(frame, det["bbox"]))

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_colors[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = distance.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col], input_colors[col])
                
        # Assign teams
        if self.team_model_trained:
            for obj_id, bbox in self.bboxes.items():
                c = self.extract_color(frame, bbox)
                team_idx = self.kmeans.predict([c])[0]
                self.player_teams[obj_id] = config.TEAM_A_LABEL if team_idx == 0 else config.TEAM_B_LABEL
        else:
            for obj_id in self.bboxes.keys():
                self.player_teams[obj_id] = "Unknown"

        return self.objects, self.bboxes, self.player_teams
