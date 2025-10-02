"""
Export training data to JSON format for cinematic visualization
"""
import json
import csv
import os
import re
from pathlib import Path


def parse_milestones(milestone_file):
    """Parse milestone text file into structured data"""
    milestones = []

    with open(milestone_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[3:]:  # Skip header lines
        line = line.strip()
        if not line:
            continue

        # Parse "Episode X: Message" format
        match = re.match(r'Episode (\d+): (.+)', line)
        if match:
            episode = int(match.group(1))
            message = match.group(2)

            # Categorize milestone types
            category = 'general'
            if 'win streak' in message.lower():
                category = 'streak'
            elif 'mastery' in message.lower() or 'excellence' in message.lower():
                category = 'mastery'
            elif 'unlocked' in message.lower():
                category = 'unlock'
            elif 'win rate' in message.lower():
                category = 'performance'
            elif 'epsilon' in message.lower():
                category = 'learning'
            elif 'first' in message.lower() and 'catch' in message.lower():
                category = 'discovery'

            milestones.append({
                'episode': episode,
                'message': message,
                'category': category
            })

    return milestones


def export_training_data(csv_file, milestone_file, output_file):
    """Export CSV and milestones to JSON"""

    # Read CSV data
    episodes = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                'episode': int(row['Episode']),
                'score': float(row['Score']),
                'success': int(row['Success']),
                'fish': row['Fish'],
                'difficulty': int(row['Difficulty']),
                'behavior': row['Behavior'],
                'episode_length': int(row['Episode_Length']),
                'epsilon': float(row['Epsilon']),
                'win_streak': int(row['Win_Streak']),
                'avg_score': float(row['Avg_Score_100']),
                'win_rate': float(row['Win_Rate_100']),
                'easy_success_rate': float(row['Easy_Success_Rate']),
                'medium_success_rate': float(row['Medium_Success_Rate']),
                'hard_success_rate': float(row['Hard_Success_Rate']),
                'shortest_catch': int(row['Shortest_Catch']),
                'behaviors_discovered': int(row['Behaviors_Discovered']),
                'sinker_rate': float(row['Sinker_Rate']),
                'dart_rate': float(row['Dart_Rate']),
                'smooth_rate': float(row['Smooth_Rate']),
                'mixed_rate': float(row['Mixed_Rate']),
                'floater_rate': float(row['Floater_Rate'])
            })

    # Parse milestones
    milestones = parse_milestones(milestone_file)

    # Combine into single JSON structure
    data = {
        'metadata': {
            'total_episodes': len(episodes),
            'total_milestones': len(milestones),
            'final_win_rate': episodes[-1]['win_rate'] if episodes else 0,
            'max_win_streak': max([e['win_streak'] for e in episodes]) if episodes else 0
        },
        'episodes': episodes,
        'milestones': milestones
    }

    # Write JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(episodes)} episodes and {len(milestones)} milestones")
    print(f"Final win rate: {data['metadata']['final_win_rate']:.1f}%")
    print(f"Max win streak: {data['metadata']['max_win_streak']}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    # Find most recent training files
    training_logs = Path("training_logs")

    csv_files = sorted(training_logs.glob("training_metrics_*.csv"))
    milestone_files = sorted(training_logs.glob("milestones_*.txt"))

    if not csv_files or not milestone_files:
        print("No training data found in training_logs/")
        exit(1)

    # Use most recent files
    latest_csv = csv_files[-1]
    latest_milestone = milestone_files[-1]

    print(f"Using training data from: {latest_csv.name}")

    # Export to visualization folder
    output_path = Path("visualization/training_data.json")
    export_training_data(latest_csv, latest_milestone, output_path)
