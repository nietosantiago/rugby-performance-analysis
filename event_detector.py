import numpy as np
import config
from models import Event

class EventDetector:
    def __init__(self):
        self.events_detected = []
        self.next_event_id = 1
        
        # State for heuristics
        self.ruck_active = False
        self.player_velocities = {} # obj_id: velocity
        self.last_centroids = {} # obj_id: (x, y)
        self.last_event_frame = 0 # To throttle events
        self.current_frame = 0
        
    def transform_to_field_coords(self, frame_shape, bbox):
        # Maps video coordinates to [0,100] x [0,70] field
        fx, fy = frame_shape[1], frame_shape[0]
        bx = (bbox[0] + bbox[2]) / 2.0
        by = float(bbox[3]) # bottom y
        
        field_x = (bx / fx) * config.FIELD_LENGTH_METERS
        field_y = (by / fy) * config.FIELD_WIDTH_METERS
        return round(field_x, 1), round(field_y, 1)

    def detect_events(self, frame_shape, minute_str, bboxes, teams):
        self.current_frame += 1
        current_centroids = {}
        
        for obj_id, bbox in bboxes.items():
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = float(bbox[3])
            current_centroids[obj_id] = (cx, cy)
            
            if obj_id in self.last_centroids:
                vx = cx - self.last_centroids[obj_id][0]
                vy = cy - self.last_centroids[obj_id][1]
                self.player_velocities[obj_id] = np.sqrt(vx**2 + vy**2)
            else:
                self.player_velocities[obj_id] = 0.0
                
        self.last_centroids = current_centroids
        
        close_pairs = []
        player_ids = list(bboxes.keys())
        for i in range(len(player_ids)):
            for j in range(i+1, len(player_ids)):
                id1 = player_ids[i]
                id2 = player_ids[j]
                
                t1 = teams.get(id1, "Unknown")
                t2 = teams.get(id2, "Unknown")
                
                c1 = current_centroids[id1]
                c2 = current_centroids[id2]
                dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                
                if dist < config.PROXIMITY_THRESHOLD_PX:
                    close_pairs.append((id1, id2, t1, t2, c1))
                    
        # Throttle to prevent spamming events every frame (1 event every 2 seconds roughly)
        if self.current_frame - self.last_event_frame < (config.PROCESS_SAMPLE_RATE_FPS * 2):
            return

        ruck_cluster = set()
        for p in close_pairs:
            ruck_cluster.add(p[0])
            ruck_cluster.add(p[1])
            
        if len(ruck_cluster) >= config.RUCK_MIN_PLAYERS:
            if not self.ruck_active:
                self.ruck_active = True
                p_id = list(ruck_cluster)[0]
                team = teams.get(p_id, config.TEAM_A_LABEL)
                x, y = self.transform_to_field_coords(frame_shape, bboxes[p_id])
                self.add_event(minute_str, team, f"Player_{p_id}", "Ruck", x, y)
        else:
            self.ruck_active = False
            tackle_detected = False
            for p in close_pairs:
                id1, id2, t1, t2, c1 = p
                if t1 != t2 and t1 != "Unknown" and t2 != "Unknown":
                    v1 = self.player_velocities.get(id1, 0)
                    v2 = self.player_velocities.get(id2, 0)
                    if v1 < 10 and v2 < 10:
                        x, y = self.transform_to_field_coords(frame_shape, bboxes[id1])
                        self.add_event(minute_str, t1, f"Player_{id1}", "Tackle", x, y)
                        tackle_detected = True
                        break
            
            if not tackle_detected:
                for obj_id, v in self.player_velocities.items():
                    if v > 15:
                        isolated = True
                        for other_id, other_c in current_centroids.items():
                            if other_id != obj_id:
                                dist = np.sqrt((current_centroids[obj_id][0]-other_c[0])**2 + (current_centroids[obj_id][1]-other_c[1])**2)
                                if dist < config.PROXIMITY_THRESHOLD_PX:
                                    isolated = False
                                    break
                        if isolated:
                            x, y = self.transform_to_field_coords(frame_shape, bboxes[obj_id])
                            team = teams.get(obj_id, config.TEAM_A_LABEL)
                            self.add_event(minute_str, team, f"Player_{obj_id}", "Carry", x, y)
                            break

    def add_event(self, minute_str, team, player, event_type, x, y):
        ev = Event(
            event_id=self.next_event_id,
            match_id="Match_1",
            minute=minute_str,
            team=team,
            player=player,
            event_type=event_type,
            x=x,
            y=y
        )
        self.events_detected.append(ev)
        self.next_event_id += 1
        self.last_event_frame = self.current_frame
