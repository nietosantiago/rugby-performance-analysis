import pandas as pd
import os
import config
from models import Event, Player, Team
from typing import List

class DataGenerator:
    def __init__(self, output_dir=config.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_events_csv(self, events: List[Event]):
        data = [{
            "event_id": e.event_id,
            "match_id": e.match_id,
            "minute": e.minute,
            "team": e.team,
            "player": e.player,
            "event_type": e.event_type,
            "x": e.x,
            "y": e.y
        } for e in events]
        df = pd.DataFrame(data)
        out_path = os.path.join(self.output_dir, "events.csv")
        df.to_csv(out_path, index=False)
        return out_path
        
    def generate_players_csv(self, players: List[Player]):
        data = [{
            "player_id": p.id,
            "team": p.team,
            "tackles": p.tackles,
            "carries": p.carries,
            "rucks": p.rucks,
            "lineouts": p.lineouts,
            "kicks": p.kicks,
            "tackle_efficiency": p.tackle_efficiency,
            "participation_index": p.participation_index,
            "impact_score": p.impact_score
        } for p in players]
        df = pd.DataFrame(data)
        out_path = os.path.join(self.output_dir, "players.csv")
        df.to_csv(out_path, index=False)
        return out_path
        
    def generate_teams_csv(self, teams: List[Team]):
        data = [{
            "team_id": t.id,
            "total_tackles": t.total_tackles,
            "total_carries": t.total_carries,
            "total_rucks": t.total_rucks,
            "total_lineouts": t.total_lineouts,
            "total_kicks": t.total_kicks,
            "territorial_dominance": t.territorial_dominance
        } for t in teams]
        df = pd.DataFrame(data)
        out_path = os.path.join(self.output_dir, "teams.csv")
        df.to_csv(out_path, index=False)
        return out_path
