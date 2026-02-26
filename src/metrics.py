"""
metrics.py — Aggregation of player and team statistics from events.

Reads ``events.csv`` (or a DataFrame) and produces ``player_stats.csv``
and ``team_stats.csv``.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MetricsAggregator:
    """Computes player-level and team-level statistics from the events list."""

    def __init__(self, events_df: pd.DataFrame):
        self.events = events_df

    # ── player stats ───────────────────────────

    def compute_player_stats(self) -> pd.DataFrame:
        """Returns a DataFrame with columns:
        player_id | team | tackles | carries | rucks | kicks | meters_gained
        """
        if self.events.empty:
            return pd.DataFrame(columns=[
                "player_id", "team", "tackles", "carries",
                "rucks", "kicks", "meters_gained",
            ])

        df = self.events.copy()
        pivot = (
            df.groupby(["player_id", "team", "event_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        stat_cols = {
            "Tackle": "tackles",
            "Carry": "carries",
            "Ruck": "rucks",
            "Kick": "kicks",
            "Lineout": "lineouts",
        }
        for orig, renamed in stat_cols.items():
            if orig not in pivot.columns:
                pivot[renamed] = 0
            else:
                pivot.rename(columns={orig: renamed}, inplace=True)

        # Estimate meters gained from carry events
        carry_events = df[df["event_type"] == "Carry"].copy()
        if not carry_events.empty:
            carry_events["x_diff"] = carry_events.groupby("player_id")["x"].diff().abs()
            meters = carry_events.groupby("player_id")["x_diff"].sum().reset_index()
            meters.columns = ["player_id", "meters_gained"]
            # Fallback: if diff gives NaN (single carry), use 3m per carry
            meters["meters_gained"] = meters["meters_gained"].fillna(3.0)
            pivot = pivot.merge(meters, on="player_id", how="left")
            pivot["meters_gained"] = pivot["meters_gained"].fillna(0.0).round(1)
        else:
            pivot["meters_gained"] = 0.0

        keep = ["player_id", "team"] + list(stat_cols.values()) + ["meters_gained"]
        keep = [c for c in keep if c in pivot.columns]
        return pivot[keep]

    # ── team stats ─────────────────────────────

    def compute_team_stats(self) -> pd.DataFrame:
        """Returns a DataFrame with columns:
        team | total_tackles | total_carries | total_possession_time
        """
        if self.events.empty:
            return pd.DataFrame(columns=[
                "team", "total_tackles", "total_carries",
                "total_rucks", "total_kicks", "total_possession_time",
            ])

        df = self.events.copy()

        team_agg = (
            df.groupby(["team", "event_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        rename_map = {
            "Tackle": "total_tackles",
            "Carry": "total_carries",
            "Ruck": "total_rucks",
            "Kick": "total_kicks",
            "Lineout": "total_lineouts",
        }
        for orig, new_name in rename_map.items():
            if orig not in team_agg.columns:
                team_agg[new_name] = 0
            else:
                team_agg.rename(columns={orig: new_name}, inplace=True)

        # Estimate possession: fraction of carry + ruck events per team
        total_events = df["event_type"].isin(["Carry", "Ruck"]).sum()
        if total_events > 0:
            poss = (
                df[df["event_type"].isin(["Carry", "Ruck"])]
                .groupby("team")
                .size()
                .reset_index(name="poss_events")
            )
            poss["total_possession_time"] = (
                poss["poss_events"] / total_events * 100
            ).round(1)
            team_agg = team_agg.merge(poss[["team", "total_possession_time"]],
                                       on="team", how="left")
            team_agg["total_possession_time"] = team_agg["total_possession_time"].fillna(0.0)
        else:
            team_agg["total_possession_time"] = 0.0

        keep = [
            "team", "total_tackles", "total_carries", "total_rucks",
            "total_kicks", "total_possession_time",
        ]
        keep = [c for c in keep if c in team_agg.columns]
        return team_agg[keep]

    # ── export ─────────────────────────────────

    def export_all(self, output_dir: str = config.PROCESSED_DIR) -> Tuple[str, str]:
        """Write player_stats.csv and team_stats.csv and return their paths."""
        os.makedirs(output_dir, exist_ok=True)

        ps = self.compute_player_stats()
        ts = self.compute_team_stats()

        ps_path = os.path.join(output_dir, "player_stats.csv")
        ts_path = os.path.join(output_dir, "team_stats.csv")

        ps.to_csv(ps_path, index=False)
        ts.to_csv(ts_path, index=False)

        return ps_path, ts_path
