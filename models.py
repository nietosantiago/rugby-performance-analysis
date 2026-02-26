from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Raw detection result from YOLO."""
    bbox: List[int]            # [x1, y1, x2, y2]
    confidence: float
    class_name: str            # "person" | "ball"
    color_hsv: Optional[np.ndarray] = None


@dataclass
class TrackedObject:
    """A tracked object with persistent ID."""
    track_id: int
    bbox: List[int]
    centroid: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)   # (vx, vy) px/frame
    speed: float = 0.0                             # magnitude px/frame
    team: str = "Unknown"
    field_pos: Tuple[float, float] = (0.0, 0.0)   # normalised (0-100)


@dataclass
class Event:
    """A detected match event."""
    event_id: int
    match_id: str
    frame: int
    event_type: str            # Tackle | Carry | Ruck | Kick | Lineout
    team: str
    player_id: str
    x: float
    y: float
    timestamp: str             # MM:SS


@dataclass
class Player:
    """Aggregated player statistics."""
    player_id: str
    team: str
    tackles: int = 0
    carries: int = 0
    rucks: int = 0
    kicks: int = 0
    meters_gained: float = 0.0
    missed_tackles: int = 0
    rucks_won: int = 0
    lineouts: int = 0

    @property
    def tackle_efficiency(self) -> float:
        total = self.tackles + self.missed_tackles
        return round(self.tackles / total, 2) if total > 0 else 0.0

    @property
    def participation_index(self) -> int:
        return self.tackles + self.carries + self.rucks

    @property
    def impact_score(self) -> float:
        return round(
            self.tackles
            + (self.carries * 1.5)
            + (self.lineouts * 2)
            + (self.kicks * 1.2),
            2,
        )


@dataclass
class Team:
    """Aggregated team statistics."""
    team: str
    total_tackles: int = 0
    total_carries: int = 0
    total_rucks: int = 0
    total_lineouts: int = 0
    total_kicks: int = 0
    total_possession_time: float = 0.0   # seconds
    territorial_dominance: float = 0.0
