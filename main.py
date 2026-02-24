import argparse
import sys
import random
import config
from video_processor import VideoProcessor
from player_detector import PlayerDetector
from player_tracker import PlayerTracker
from event_detector import EventDetector
from data_generator import DataGenerator
from metrics import MetricsCalculator
from visualizer import Visualizer
from models import Event

def run_pipeline(video_path: str):
    print(f"Starting analysis on {video_path}...")
    processor = VideoProcessor(video_path)
    detector = PlayerDetector()
    tracker = PlayerTracker()
    event_det = EventDetector()
    
    frame_gen = processor.get_frames()
    
    # Process frames
    for current_frame, minute_str, frame in frame_gen:
        if current_frame % 30 == 0:
            print(f"Processing minute {minute_str} (Frame {current_frame}/{processor.total_frames})")
            
        detections = detector.detect(frame)
        objects, bboxes, teams = tracker.update(frame, detections)
        event_det.detect_events(frame.shape, minute_str, bboxes, teams)
        
    print("Video processing complete.")
    return event_det.events_detected

def generate_demo_data():
    print("Running in DEMO mode. Generating synthetic events...")
    events = []
    teams = [config.TEAM_A_LABEL, config.TEAM_B_LABEL]
    event_types = ["Tackle", "Carry", "Ruck", "Lineout", "Kick"]
    
    for i in range(1, 51):
        minute = f"{i//5:02d}:{random.randint(0, 59):02d}"
        team = random.choice(teams)
        player = f"Player_{random.randint(1, 15)}"
        ev_type = random.choices(event_types, weights=[40, 30, 15, 10, 5])[0]
        x = random.uniform(5.0, 95.0)
        y = random.uniform(5.0, 65.0)
        
        # Add some structure: Team B tackles happen more on Team A's side
        if ev_type == "Tackle" and team == config.TEAM_B_LABEL:
            x = random.uniform(5.0, 50.0)
        elif ev_type == "Tackle" and team == config.TEAM_A_LABEL:
            x = random.uniform(50.0, 95.0)
            
        events.append(Event(
            event_id=i, match_id="Demo_Match_1", minute=minute,
            team=team, player=player, event_type=ev_type, x=round(x,1), y=round(y,1)
        ))
    return events

def main():
    parser = argparse.ArgumentParser(description="Rugby Video Analysis System")
    parser.add_argument("--video", type=str, help="Path to the match video file")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data for testing")
    parser.add_argument("--sample-rate", type=int, default=config.PROCESS_SAMPLE_RATE_FPS, 
                        help="Frames per second to process (lower is faster)")
    args = parser.parse_args()

    config.PROCESS_SAMPLE_RATE_FPS = args.sample_rate

    if args.demo:
        events = generate_demo_data()
    elif args.video:
        try:
            events = run_pipeline(args.video)
        except Exception as e:
            print(f"Error processing video: {e}")
            sys.exit(1)
    else:
        print("Please provide a --video path or use --demo.")
        parser.print_help()
        sys.exit(1)
        
    if not events:
        print("No events detected. Exiting.")
        sys.exit(0)
        
    print(f"Detected {len(events)} events. Calculating metrics...")
    
    # Calculate metrics
    calc = MetricsCalculator(events)
    players, teams = calc.calculate()
    insights = calc.get_insights()
    
    print("\n--- Match Insights ---")
    print(f"Most Active Player: {insights['most_active_player']}")
    print(f"Dominant Team: {insights['dominant_team']}")
    print(f"Most Frequent Zone: {insights['frequent_zone']}")
    print("----------------------\n")
    
    # Generate Outputs
    print("Generating CSV datasets...")
    data_gen = DataGenerator()
    data_gen.generate_events_csv(events)
    data_gen.generate_players_csv(players)
    data_gen.generate_teams_csv(teams)
    
    print("Generating Visualizations...")
    viz = Visualizer()
    viz.generate_all_heatmaps(events)
    viz.print_timeline(events)
    
    print(f"\nAll done! Outputs saved to ./{config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
