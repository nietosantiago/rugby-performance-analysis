"""
heatmaps.py — Kernel Density Estimation (KDE) heatmap generation.

Generates spatial heatmaps by team, player, and event type.
Outputs PNGs to reports/figures/ and coordinate CSVs for dynamic rendering.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class HeatmapGenerator:
    """Generates KDE heatmaps with Gaussian distribution over a rugby pitch."""

    def __init__(
        self,
        output_dir: str = config.FIGURES_DIR,
        field_length: float = config.FIELD_LENGTH,
        field_width: float = config.FIELD_WIDTH,
    ):
        self.output_dir = output_dir
        self.fl = field_length
        self.fw = field_width
        os.makedirs(output_dir, exist_ok=True)

    # ── pitch drawing ─────────────────────────

    def _draw_field(self, ax: plt.Axes) -> None:
        """Draw a rugby pitch background with key lines."""
        ax.set_facecolor("#2d5a27")
        ax.set_xlim(0, self.fl)
        ax.set_ylim(0, self.fw)

        # Boundary
        ax.plot(
            [0, self.fl, self.fl, 0, 0],
            [0, 0, self.fw, self.fw, 0],
            color="white", linewidth=2,
        )
        # Halfway line
        ax.axvline(self.fl / 2, color="white", linestyle="--", alpha=0.7)
        # 22-metre lines
        ax.axvline(22, color="white", linestyle=":", alpha=0.5)
        ax.axvline(self.fl - 22, color="white", linestyle=":", alpha=0.5)
        # Try lines
        ax.axvline(5, color="white", linestyle=":", alpha=0.3)
        ax.axvline(self.fl - 5, color="white", linestyle=":", alpha=0.3)

    # ── core generator ─────────────────────────

    def generate(
        self,
        coords: List[Tuple[float, float]],
        title: str,
        filename: str,
        cmap: str = "YlOrRd",
        bw_adjust: float = 0.5,
    ) -> Optional[str]:
        """Generate a KDE heatmap from ``(x, y)`` coordinate pairs.

        Returns the path to the saved PNG, or *None* if too few points.
        """
        if len(coords) < 2:
            return None

        x = [c[0] for c in coords]
        y = [c[1] for c in coords]

        fig, ax = plt.subplots(figsize=(12, 8))
        self._draw_field(ax)

        if len(coords) > 3:
            sns.kdeplot(
                x=x, y=y, cmap=cmap, fill=True,
                alpha=0.6, ax=ax, bw_adjust=bw_adjust,
                levels=20,
            )
        else:
            ax.scatter(x, y, c="red", s=120, zorder=5, edgecolors="white")

        ax.set_title(title, fontsize=14, fontweight="bold", color="white")
        ax.set_xlabel("Largo de Cancha (m)", color="white")
        ax.set_ylabel("Ancho de Cancha (m)", color="white")
        ax.tick_params(colors="white")

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return path

    # ── batch generators ───────────────────────

    def generate_by_team(self, events_df: pd.DataFrame) -> List[str]:
        """Generate one heatmap per team × event-type combination."""
        paths: list[str] = []
        for team in events_df["team"].unique():
            if team == "Unknown":
                continue
            for etype in events_df["event_type"].unique():
                subset = events_df[
                    (events_df["team"] == team)
                    & (events_df["event_type"] == etype)
                ]
                coords = list(zip(subset["x"], subset["y"]))
                p = self.generate(
                    coords,
                    title=f"{etype} — {team}",
                    filename=f"heatmap_{etype}_{team}.png",
                )
                if p:
                    paths.append(p)
        return paths

    def generate_by_player(
        self, events_df: pd.DataFrame, player_id: str
    ) -> Optional[str]:
        """Generate a heatmap for a single player."""
        subset = events_df[events_df["player_id"] == player_id]
        coords = list(zip(subset["x"], subset["y"]))
        return self.generate(
            coords,
            title=f"Actividad — {player_id}",
            filename=f"heatmap_player_{player_id}.png",
        )

    # ── CSV export for dashboard ───────────────

    @staticmethod
    def export_coordinates_csv(
        events_df: pd.DataFrame, output_path: str
    ) -> None:
        """Export aggregated coordinates for dynamic rendering in the dashboard."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        agg = (
            events_df.groupby(["team", "event_type"])
            .agg(
                x_mean=("x", "mean"),
                y_mean=("y", "mean"),
                count=("x", "count"),
            )
            .reset_index()
        )
        agg.to_csv(output_path, index=False)
