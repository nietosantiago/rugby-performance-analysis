"""
video_processor.py — Main pipeline orchestrator.

Reads a video file frame-by-frame (with configurable frame-skipping),
runs detection → tracking → team classification → event detection,
and returns an events DataFrame.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.detection import BallDetector, PlayerDetector, TeamClassifier
from src.tracking import BallTracker, PlayerTracker
from src.events_logic import EventEngine, FieldMapper


class VideoProcessor:
    """End-to-end pipeline for a single match video."""

    def __init__(self):
        # Detection
        self.player_detector = PlayerDetector()
        self.ball_detector = BallDetector()
        self.team_classifier = TeamClassifier()

        # Tracking
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()

        # Events
        self.field_mapper = FieldMapper()
        self.event_engine = EventEngine(self.field_mapper)

    # ── public API ─────────────────────────────

    def process(
        self,
        video_path: str,
        match_id: str = "Match_1",
        frame_skip: Optional[int] = config.FRAME_SKIP,
    ) -> pd.DataFrame:
        """Process *video_path* and return an events DataFrame.

        Parameters
        ----------
        video_path : str
            Path to the .mp4 (or other video format).
        match_id : str
            Identifier written into every event row.
        frame_skip : int | None
            Process every *n*-th frame.  ``None`` → auto from fps.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = frame_skip or max(1, int(fps / config.PROCESS_SAMPLE_RATE_FPS))

        print(f"  Video   : {video_path}")
        print(f"  FPS     : {fps:.1f}")
        print(f"  Frames  : {total_frames}")
        print(f"  Skip    : every {skip} frames")

        t0 = time.time()
        frame_num = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % skip != 0:
                frame_num += 1
                continue

            timestamp = self._format_ts(frame_num, fps)

            # ── 1. Detection ──────────────────
            player_dets = self.player_detector.detect(frame)
            ball_det = self.ball_detector.detect(frame)

            # ── 2. Tracking ───────────────────
            tracked_players = self.player_tracker.update(player_dets, frame)
            ball_pos = self.ball_tracker.update(ball_det)

            # ── 3. Team classification ────────
            for tp in tracked_players:
                hsv = self.team_classifier.extract_jersey_hsv(frame, tp.bbox)
                self.team_classifier.add_sample(hsv)
                tp.team = self.team_classifier.classify(hsv)
                tp.field_pos = self.field_mapper.pixel_to_field(
                    tp.centroid, frame.shape
                )

            # ── 4. Event detection ────────────
            self.event_engine.process_frame(
                tracked_players=tracked_players,
                ball_pos=ball_pos,
                ball_speed=self.ball_tracker.ball_speed,
                frame_num=frame_num,
                frame_shape=frame.shape,
                timestamp=timestamp,
                match_id=match_id,
            )

            frame_num += 1
            processed += 1

            if processed % 50 == 0:
                pct = (frame_num / max(total_frames, 1)) * 100
                print(f"  Progreso: {pct:.1f}%  ({len(self.event_engine.events)} eventos)")

        cap.release()
        elapsed = time.time() - t0

        print(f"\n  Procesamiento completado en {elapsed:.1f}s")
        print(f"  Frames procesados : {processed}")
        print(f"  Eventos detectados: {len(self.event_engine.events)}")

        # Build DataFrame
        if not self.event_engine.events:
            return pd.DataFrame(columns=[
                "event_id", "match_id", "frame", "event_type",
                "team", "player_id", "x", "y", "timestamp",
            ])

        rows = [
            {
                "event_id": e.event_id,
                "match_id": e.match_id,
                "frame": e.frame,
                "event_type": e.event_type,
                "team": e.team,
                "player_id": e.player_id,
                "x": e.x,
                "y": e.y,
                "timestamp": e.timestamp,
            }
            for e in self.event_engine.events
        ]
        return pd.DataFrame(rows)

    # ── helpers ────────────────────────────────

    @staticmethod
    def _format_ts(frame: int, fps: float) -> str:
        secs = frame / fps
        m = int(secs // 60)
        s = int(secs % 60)
        return f"{m:02d}:{s:02d}"


# ───────────────────────────────────────────────
# Convenience function (backward compatible)
# ───────────────────────────────────────────────

def process_video(video_folder: str) -> pd.DataFrame:
    """Process all .mp4 files in *video_folder* and return a combined events DF."""
    all_events = []
    processor = VideoProcessor()

    for fname in sorted(os.listdir(video_folder)):
        if not fname.lower().endswith(".mp4"):
            continue
        print(f"\nProcesando: {fname}")
        vpath = os.path.join(video_folder, fname)
        df = processor.process(vpath, match_id=os.path.splitext(fname)[0])
        all_events.append(df)

    if all_events:
        return pd.concat(all_events, ignore_index=True)
    return pd.DataFrame()
