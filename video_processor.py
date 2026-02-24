import cv2
import config

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0 # fallback
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.target_fps = config.PROCESS_SAMPLE_RATE_FPS
        self.frame_interval = max(1, int(self.fps / self.target_fps))
        
    def get_frames(self):
        """Yields (frame_number, minute_str, frame) at the configured sample rate."""
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
