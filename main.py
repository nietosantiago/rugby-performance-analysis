"""
main.py — Entry point for the rugby video analysis pipeline.

Usage:
    python main.py
"""

import os
import sys
import time

import pandas as pd
import yaml

import config
from src.video_processor import process_video
from src.metrics import MetricsAggregator
from src.heatmaps import HeatmapGenerator


def main() -> None:
    # ── Load YAML config ──────────────────────
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    video_folder = cfg["data"]["videos_folder"]

    # Ensure output directories exist
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    t0 = time.time()

    # ── 1. Process videos ─────────────────────
    print("=" * 50)
    print("  RUGBY VIDEO ANALYSIS PIPELINE")
    print("=" * 50)
    print(f"\nBuscando videos en: {video_folder}")

    events_df = process_video(video_folder)

    if events_df.empty:
        print("\n⚠  No se detectaron eventos. Verifica los videos de entrada.")
        return

    # ── 2. Save events.csv ────────────────────
    events_path = os.path.join(config.PROCESSED_DIR, "events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"\n✓ Eventos guardados en: {events_path}  ({len(events_df)} filas)")

    # Also save to legacy output/ for backward compat
    events_df.to_csv(os.path.join(config.OUTPUT_DIR, "events.csv"), index=False)

    # ── 3. Compute metrics ────────────────────
    print("\nCalculando métricas...")
    aggregator = MetricsAggregator(events_df)
    ps_path, ts_path = aggregator.export_all(config.PROCESSED_DIR)
    print(f"✓ Player stats: {ps_path}")
    print(f"✓ Team stats  : {ts_path}")

    # Legacy copies
    aggregator.export_all(config.OUTPUT_DIR)

    # ── 4. Generate heatmaps ──────────────────
    print("\nGenerando heatmaps...")
    hm = HeatmapGenerator(output_dir=config.FIGURES_DIR)
    paths = hm.generate_by_team(events_df)
    for p in paths:
        print(f"  ✓ {p}")

    # Export coordinate CSV for dashboard
    coord_csv = os.path.join(config.PROCESSED_DIR, "heatmap_coords.csv")
    HeatmapGenerator.export_coordinates_csv(events_df, coord_csv)
    print(f"✓ Coordenadas para dashboard: {coord_csv}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  Pipeline completado en {elapsed:.1f}s")
    print(f"  Eventos : {len(events_df)}")
    print(f"  Equipos : {events_df['team'].nunique()}")
    print(f"  Jugadores: {events_df['player_id'].nunique()}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()