import pandas as pd
import os
import config
from models import Event, Player, Team
from typing import List, Dict
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
            "meters_gained": p.meters_gained,
            "rucks": p.rucks,
            "rucks_won": p.rucks_won,
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
    
    def generate_metrics_csv(self, players: List[Player], teams: List[Team]):
        data = []
        for p in players:
            rucks_won_pct = round((p.rucks_won / p.rucks) * 100, 2) if p.rucks > 0 else 0.0
            avg_meters_per_carry = round(p.meters_gained / p.carries, 2) if p.carries > 0 else 0.0
            
            data.append({
                "entity_type": "player",
                "id": p.id,
                "team": p.team,
                "tackle_efficiency_pct": p.tackle_efficiency * 100,
                "meters_gained_per_carry": avg_meters_per_carry,
                "rucks_won_pct": rucks_won_pct,
                "participation_index": p.participation_index
            })
            
        for t in teams:
             data.append({
                "entity_type": "team",
                "id": t.id,
                "team": t.id,
                "tackle_efficiency_pct": 0.0, # Will calculate globally if needed or ignore
                "meters_gained_per_carry": 0.0,
                "rucks_won_pct": 0.0,
                 "participation_index": 0
            })
             
        df = pd.DataFrame(data)
        out_path = os.path.join(self.output_dir, "metrics.csv")
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
