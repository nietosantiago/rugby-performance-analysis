import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import config
from models import Event
from typing import List

class Visualizer:
    def __init__(self, output_dir=config.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_heatmap(self, events: List[Event], event_type: str, team: str, filename: str):
        filtered = [e for e in events if e.event_type == event_type and e.team == team]
        if not filtered:
            print(f"No {event_type} events for {team} to plot.")
            return None
            
        x = [e.x for e in filtered]
        y = [e.y for e in filtered]
        
        plt.figure(figsize=(10, 7))
        # Draw a basic rugby field outline
        plt.plot([0, 100, 100, 0, 0], [0, 0, 70, 70, 0], color="white", linewidth=2)
        plt.plot([50, 50], [0, 70], color="white", linestyle="--") # Halfway line
        plt.plot([22, 22], [0, 70], color="white", linestyle=":") # 22m line left
        plt.plot([78, 78], [0, 70], color="white", linestyle=":") # 22m line right
        
        ax = plt.gca()
        ax.set_facecolor('#4CAF50') # Grass green color
        
        # Avoid KDE errors if too few points
        if len(x) > 2:
            sns.kdeplot(x=x, y=y, cmap="Reds", fill=True, alpha=0.6, ax=ax, bw_adjust=0.5)
        else:
            sns.scatterplot(x=x, y=y, color="red", s=100, ax=ax)
        
        plt.xlim(0, 100)
        plt.ylim(0, 70)
        plt.title(f"{event_type} Heatmap - {team}")
        plt.xlabel("Field Length (m)")
        plt.ylabel("Field Width (m)")
        
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        return out_path
        
    def generate_all_heatmaps(self, events: List[Event]):
        teams = set(e.team for e in events if e.team != "Unknown")
        for t in teams:
            self.generate_heatmap(events, "Tackle", t, f"heatmap_tackles_{t}.png")
            self.generate_heatmap(events, "Carry", t, f"heatmap_carries_{t}.png")
            
    def print_timeline(self, events: List[Event]):
        timeline = []
        for e in sorted(events, key=lambda x: x.event_id):
            timeline.append(f"{e.minute} {e.event_type} {e.player} ({e.team})")
            
        out_path = os.path.join(self.output_dir, "timeline.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(timeline))
            
        return out_path
