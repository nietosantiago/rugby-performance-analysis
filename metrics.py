import pandas as pd
from models import Event, Player, Team
from typing import List, Dict, Tuple

class MetricsCalculator:
    def __init__(self, events: List[Event]):
        self.events = events
        self.players: Dict[str, Player] = {}
        self.teams: Dict[str, Team] = {}
        
    def calculate(self) -> Tuple[List[Player], List[Team]]:
        for ev in self.events:
            # Initialize Team
            if ev.team not in self.teams:
                self.teams[ev.team] = Team(id=ev.team)
                
            # Initialize Player
            if ev.player not in self.players:
                self.players[ev.player] = Player(id=ev.player, team=ev.team)
                
            p = self.players[ev.player]
            t = self.teams[ev.team]
            
            # Aggregate events
            if ev.event_type == "Tackle":
                p.tackles += 1
                t.total_tackles += 1
            elif ev.event_type == "Carry":
                p.carries += 1
                t.total_carries += 1
            elif ev.event_type == "Ruck":
                p.rucks += 1
                t.total_rucks += 1
            elif ev.event_type == "Lineout":
                p.lineouts += 1
                t.total_lineouts += 1
            elif ev.event_type == "Kick":
                p.kicks += 1
                t.total_kicks += 1
                
        # Calculate Team Territorial Dominance (Average X coordinate)
        team_x_coords = {}
        for ev in self.events:
            if ev.team not in team_x_coords:
                team_x_coords[ev.team] = []
            team_x_coords[ev.team].append(ev.x)
            
        for team_id, x_list in team_x_coords.items():
            if len(x_list) > 0:
                self.teams[team_id].territorial_dominance = round(sum(x_list) / len(x_list), 2)
                
        return list(self.players.values()), list(self.teams.values())
        
    def get_insights(self) -> dict:
        players_list = list(self.players.values())
        teams_list = list(self.teams.values())
        
        if not players_list or not teams_list:
            return {
                "most_active_player": "N/A",
                "dominant_team": "N/A",
                "frequent_zone": "N/A"
            }
            
        most_active = max(players_list, key=lambda p: p.participation_index)
        dominant_team = max(teams_list, key=lambda t: t.territorial_dominance)
        
        # Most frequent zone (binning X coords)
        x_coords = [e.x for e in self.events]
        if x_coords:
            df = pd.DataFrame({"x": x_coords})
            zone = pd.cut(df['x'], bins=[0, 22, 50, 78, 100], labels=["Own 22", "Midfield (Own)", "Midfield (Opp)", "Opp 22"]).mode()[0]
        else:
            zone = "Unknown"
            
        return {
            "most_active_player": most_active.id,
            "dominant_team": dominant_team.id,
            "frequent_zone": zone
        }
