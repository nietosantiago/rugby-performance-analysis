import cv2
import numpy as np
import config
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL_NAME)
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Detects persons in the given frame.
        Returns a list of dicts with bounding boxes and confidence.
        """
        # We only want to detect class 0 (person)
        results = self.model(frame, classes=[0], conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if hasattr(boxes, 'xyxy') and len(boxes.xyxy) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    # Filter by minimum area to ignore noise or small background detections
                    area = (x2 - x1) * (y2 - y1)
                    if area >= config.MIN_BBOX_AREA:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf
                        })
                        
        return detections
