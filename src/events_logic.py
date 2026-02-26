"""
events_logic.py — Rule-based event detection engine.

Individual detector classes for each rugby event type, orchestrated by
EventEngine which runs all detectors on every processed frame.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models import Event, TrackedObject


# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────

class FieldMapper:
    """Converts pixel coordinates to normalised field coordinates [0-100]."""

    def __init__(
        self,
        field_length: float = config.FIELD_LENGTH,
        field_width: float = config.FIELD_WIDTH,
    ):
        self.fl = field_length
        self.fw = field_width

    def pixel_to_field(
        self, centroid: Tuple[float, float], frame_shape: tuple
    ) -> Tuple[float, float]:
        frame_h, frame_w = frame_shape[:2]
        x = (centroid[0] / frame_w) * self.fl
        y = (centroid[1] / frame_h) * self.fw
        return (round(x, 1), round(y, 1))


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _find_nearest_player(
    players: List[TrackedObject], point: Tuple[float, float]
) -> Optional[TrackedObject]:
    if not players or point is None:
        return None
    best, best_d = None, float("inf")
    for p in players:
        d = _distance(p.centroid, point)
        if d < best_d:
            best_d = d
            best = p
    return best


def _majority_team(players: List[TrackedObject]) -> str:
    teams = [p.team for p in players if p.team != "Unknown"]
    if not teams:
        return "Unknown"
    from collections import Counter
    return Counter(teams).most_common(1)[0][0]


# ───────────────────────────────────────────────
# Event Detectors
# ───────────────────────────────────────────────

class TackleDetector:
    """Two players from opposite teams converge AND the ball-carrier
    decelerates sharply (speed drops > 60 %)."""

    def __init__(self):
        self.last_frame = -999
        self.prev_speeds: dict[int, float] = {}

    def detect(
        self,
        players: List[TrackedObject],
        ball_pos: Optional[Tuple[float, float]],
        frame_num: int,
    ) -> Optional[dict]:
        if frame_num - self.last_frame < config.TACKLE_COOLDOWN:
            return None

        carrier = _find_nearest_player(players, ball_pos)
        if carrier is None or carrier.team == "Unknown":
            self._update_speeds(players)
            return None

        prev = self.prev_speeds.get(carrier.track_id, carrier.speed)

        for p in players:
            if p.team == carrier.team or p.team == "Unknown":
                continue
            if _distance(p.centroid, carrier.centroid) < config.PROXIMITY_THRESHOLD_PX:
                if prev > 0 and carrier.speed < prev * 0.4:
                    self.last_frame = frame_num
                    self._update_speeds(players)
                    return {
                        "event_type": "Tackle",
                        "player": carrier,
                        "tackler": p,
                    }

        self._update_speeds(players)
        return None

    def _update_speeds(self, players: List[TrackedObject]) -> None:
        for p in players:
            self.prev_speeds[p.track_id] = p.speed


class CarryDetector:
    """Ball-carrier advances > CARRY_METERS_THRESHOLD in field-metres."""

    def __init__(self):
        self.carry_start: dict[int, Tuple[float, float]] = {}

    def detect(
        self,
        players: List[TrackedObject],
        ball_pos: Optional[Tuple[float, float]],
        frame_num: int,
    ) -> Optional[dict]:
        carrier = _find_nearest_player(players, ball_pos)
        if carrier is None or carrier.team == "Unknown":
            return None

        tid = carrier.track_id

        if tid not in self.carry_start:
            self.carry_start[tid] = carrier.field_pos
            return None

        start = self.carry_start[tid]
        metres = _distance(start, carrier.field_pos)

        if metres >= config.CARRY_METERS_THRESHOLD:
            self.carry_start.pop(tid, None)
            return {
                "event_type": "Carry",
                "player": carrier,
                "meters_gained": round(metres, 1),
            }

        return None


class RuckDetector:
    """3+ players clustered within RUCK_CLUSTER_RADIUS_PX of the ball."""

    def __init__(self):
        self.ruck_active = False
        self.last_frame = -999

    def detect(
        self,
        players: List[TrackedObject],
        ball_pos: Optional[Tuple[float, float]],
        frame_num: int,
    ) -> Optional[dict]:
        if ball_pos is None:
            self.ruck_active = False
            return None

        nearby = [
            p
            for p in players
            if _distance(p.centroid, ball_pos) < config.RUCK_CLUSTER_RADIUS_PX
        ]

        if len(nearby) >= config.RUCK_MIN_PLAYERS:
            if not self.ruck_active:
                self.ruck_active = True
                if frame_num - self.last_frame > config.TACKLE_COOLDOWN:
                    self.last_frame = frame_num
                    return {
                        "event_type": "Ruck",
                        "team": _majority_team(nearby),
                        "players_involved": len(nearby),
                    }
        else:
            self.ruck_active = False

        return None


class KickDetector:
    """Ball speed spikes > KICK_SPEED_MULTIPLIER × previous speed."""

    def __init__(self):
        self.prev_speed = 0.0
        self.last_frame = -999

    def detect(
        self,
        players: List[TrackedObject],
        ball_speed: float,
        ball_pos: Optional[Tuple[float, float]],
        frame_num: int,
    ) -> Optional[dict]:
        result = None

        if (
            ball_speed > self.prev_speed * config.KICK_SPEED_MULTIPLIER
            and ball_speed > config.MIN_KICK_SPEED
            and frame_num - self.last_frame > config.KICK_COOLDOWN
        ):
            kicker = _find_nearest_player(players, ball_pos)
            if kicker is not None:
                self.last_frame = frame_num
                result = {
                    "event_type": "Kick",
                    "player": kicker,
                }

        self.prev_speed = ball_speed
        return result


class LineoutDetector:
    """4+ players aligned vertically (low X variance) near a sideline."""

    def __init__(self):
        self.last_frame = -999

    def detect(
        self,
        players: List[TrackedObject],
        frame_shape: tuple,
        frame_num: int,
    ) -> Optional[dict]:
        if frame_num - self.last_frame < config.LINEOUT_COOLDOWN:
            return None

        thr = config.LINEOUT_LATERAL_THRESHOLD
        lateral = [
            p
            for p in players
            if p.field_pos[1] < thr or p.field_pos[1] > (100 - thr)
        ]

        if len(lateral) < config.LINEOUT_MIN_PLAYERS:
            return None

        x_coords = [p.field_pos[0] for p in lateral]
        if np.std(x_coords) < 5.0:  # low spread in X → vertical alignment
            self.last_frame = frame_num
            return {
                "event_type": "Lineout",
                "team": _majority_team(lateral),
            }

        return None


# ───────────────────────────────────────────────
# Event Engine (orchestrator)
# ───────────────────────────────────────────────

class EventEngine:
    """Runs all event detectors on each processed frame."""

    def __init__(self, field_mapper: FieldMapper):
        self.field_mapper = field_mapper
        self.tackle_det = TackleDetector()
        self.carry_det = CarryDetector()
        self.ruck_det = RuckDetector()
        self.kick_det = KickDetector()
        self.lineout_det = LineoutDetector()
        self.events: List[Event] = []
        self._next_id = 1

    def process_frame(
        self,
        tracked_players: List[TrackedObject],
        ball_pos: Optional[Tuple[float, float]],
        ball_speed: float,
        frame_num: int,
        frame_shape: tuple,
        timestamp: str,
        match_id: str = "Match_1",
    ) -> List[Event]:
        new_events: List[Event] = []

        # Tackle
        res = self.tackle_det.detect(tracked_players, ball_pos, frame_num)
        if res:
            p = res["player"]
            new_events.append(self._make_event(
                res["event_type"], p.team, f"Player_{p.track_id}",
                p.field_pos, frame_num, timestamp, match_id,
            ))

        # Carry
        res = self.carry_det.detect(tracked_players, ball_pos, frame_num)
        if res:
            p = res["player"]
            new_events.append(self._make_event(
                res["event_type"], p.team, f"Player_{p.track_id}",
                p.field_pos, frame_num, timestamp, match_id,
            ))

        # Ruck
        res = self.ruck_det.detect(tracked_players, ball_pos, frame_num)
        if res:
            new_events.append(self._make_event(
                res["event_type"], res["team"], "Multiple",
                self.field_mapper.pixel_to_field(ball_pos, frame_shape)
                if ball_pos else (50.0, 50.0),
                frame_num, timestamp, match_id,
            ))

        # Kick
        res = self.kick_det.detect(
            tracked_players, ball_speed, ball_pos, frame_num
        )
        if res:
            p = res["player"]
            new_events.append(self._make_event(
                res["event_type"], p.team, f"Player_{p.track_id}",
                p.field_pos, frame_num, timestamp, match_id,
            ))

        # Lineout
        res = self.lineout_det.detect(tracked_players, frame_shape, frame_num)
        if res:
            new_events.append(self._make_event(
                res["event_type"], res["team"], "Multiple",
                (50.0, 5.0), frame_num, timestamp, match_id,
            ))

        self.events.extend(new_events)
        return new_events

    def _make_event(
        self,
        event_type: str,
        team: str,
        player_id: str,
        field_pos: Tuple[float, float],
        frame_num: int,
        timestamp: str,
        match_id: str,
    ) -> Event:
        ev = Event(
            event_id=self._next_id,
            match_id=match_id,
            frame=frame_num,
            event_type=event_type,
            team=team,
            player_id=player_id,
            x=field_pos[0],
            y=field_pos[1],
            timestamp=timestamp,
        )
        self._next_id += 1
        return ev
