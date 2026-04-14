import cv2
import config
import os
from models import Event


class VideoProcessor:
    def __init__(self, video_path: str):

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.target_fps = config.PROCESS_SAMPLE_RATE_FPS
        self.frame_interval = max(1, int(self.fps / self.target_fps))


    def get_frames(self):

        current_frame = 0

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if not ret:
                break

            if current_frame % self.frame_interval == 0:

                seconds_elapsed = current_frame / self.fps

                minutes = int(seconds_elapsed // 60)
                seconds = int(seconds_elapsed % 60)

                minute_str = f"{minutes:02d}:{seconds:02d}"

                yield current_frame, minute_str, frame

            current_frame += 1

        self.cap.release()



def process_video(video_folder):

    from player_detector import PlayerDetector
    from player_tracker import PlayerTracker
    from event_detector import EventDetector

    print("Videos encontrados:")
    events = []

    detector = PlayerDetector()

    # 🔥 CONFIGURACIÓN DE PERFORMANCE
    TARGET_WIDTH = 960
    TARGET_HEIGHT = 540

    MAX_FRAMES_TEST = None   # 👉 poner 2000 para modo test rápido

    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):

            print("Procesando:", file)

            video_path = os.path.join(video_folder, file)
            processor = VideoProcessor(video_path)

            tracker = PlayerTracker()
            event_detector = EventDetector()

            processed_frames = 0

            for frame_number, minute, frame in processor.get_frames():

                # 🔥 Limitar cantidad de frames en modo test
                if MAX_FRAMES_TEST is not None and processed_frames >= MAX_FRAMES_TEST:
                    break

                # 🔥 Reducir resolución (GRAN impacto en CPU)
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                detections = detector.detect(frame)
                objects, bboxes, teams = tracker.update(frame, detections)

                event_detector.detect_events(frame.shape, minute, bboxes, teams)

                processed_frames += 1

            events.extend(event_detector.events_detected)

    print("Procesamiento terminado")
    return events