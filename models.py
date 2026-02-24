from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Player:
    id: str
    team: str
    tackles: int = 0
    carries: int = 0
    rucks: int = 0
    lineouts: int = 0
    kicks: int = 0
    
    missed_tackles: int = 0

    @property
    def tackle_efficiency(self) -> float:
        total = self.tackles + self.missed_tackles
        return round(self.tackles / total, 2) if total > 0 else 0.0

    @property
    def participation_index(self) -> int:
        return self.tackles + self.carries + self.rucks

    @property
    def impact_score(self) -> float:
        return round(self.tackles + (self.carries * 1.5) + (self.lineouts * 2) + (self.kicks * 1.2), 2)


@dataclass
class Event:
    event_id: int
    match_id: str
    minute: str  # Format MM:SS
    team: str
    player: str
    event_type: str  # "Tackle", "Carry", "Ruck", "Lineout", "Kick"
    x: float
    y: float

@dataclass
class Team:
    id: str
    total_tackles: int = 0
    total_carries: int = 0
    total_rucks: int = 0
    total_lineouts: int = 0
    total_kicks: int = 0
    territorial_dominance: float = 0.0
