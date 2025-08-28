"""
Visualize activity classification results from SSL+HMM inference

Usage:
    python visualize_activity.py /path/to/7111.csv
    python visualize_activity.py /path/to/7111.csv --days 3  # Show only first 3 days
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse
import numpy as np

def load_activity_data(csv_file):
    """Load activity classification results"""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df

def plot_activity_timeline(df, participant_id, days=None, save_path=None):
    """Create timeline visualization of activity classifications"""
    
    # Limit to specific number of days if requested
    if days:
        start_time = df.index[0]
        end_time = start_time + pd.Timedelta(days=days)
        df = df[df.index <= end_time]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Define colors for each activity
    activity_colors = {
        'sleep': '#2E86AB',      # Blue
        'sedentary': '#A23B72',  # Purple  
        'light': '#F18F01',      # Orange
        'moderate': '#C73E1D'    # Red
    }
    
    # Get unique activities in order
    activities = df['activity'].unique()
    
    # Create y-values for activities
    activity_y = {act: i for i, act in enumerate(activities)}
    
    # Plot timeline
    for activity in activities:
        mask = df['activity'] == activity
        times = df[mask].index
        y_vals = [activity_y[activity]] * len(times)
        
        ax.scatter(times, y_vals, 
                  c=activity_colors.get(activity, 'gray'), 
                  label=activity, 
                  alpha=0.7, 
                  s=1)
    
    # Formatting
    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels(activities)
    ax.set_ylabel('Activity Type')
    ax.set_xlabel('Time')
    ax.set_title(f'Activity Classification Timeline - Participant {participant_id}')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to: {save_path}")
    
    plt.show()

def plot_daily_activity_summary(df, participant_id, save_path=None):
    """Create daily activity summary bar chart"""
    
    # Add date column
    df['date'] = df.index.date
    
    # Calculate daily activity proportions
    daily_counts = df.groupby(['date', 'activity']).size().unstack(fill_value=0)
    daily_props = daily_counts.div(daily_counts.sum(axis=1), axis=0) * 100
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    activity_colors = {
        'sleep': '#2E86AB',
        'sedentary': '#A23B72', 
        'light': '#F18F01',
        'moderate': '#C73E1D'
    }
    
    # Plot stacked bars
    bottom = np.zeros(len(daily_props))
    
    for activity in daily_props.columns:
        ax.bar(daily_props.index, daily_props[activity], 
               bottom=bottom, 
               label=activity,
               color=activity_colors.get(activity, 'gray'))
        bottom += daily_props[activity]
    
    # Formatting
    ax.set_ylabel('Percentage of Day (%)')
    ax.set_xlabel('Date')
    ax.set_title(f'Daily Activity Distribution - Participant {participant_id}')
    ax.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Daily summary plot saved to: {save_path}")
    
    plt.show()

def print_activity_summary(df, participant_id):
    """Print summary statistics"""
    print(f"\n=== Activity Summary for Participant {participant_id} ===")
    print(f"Data period: {df.index[0]} to {df.index[-1]}")
    print(f"Total duration: {df.index[-1] - df.index[0]}")
    print(f"Total data points: {len(df):,}")
    
    # Activity distribution
    activity_counts = df['activity'].value_counts()
    activity_pcts = (activity_counts / len(df) * 100).round(1)
    
    print(f"\nActivity Distribution:")
    for activity, pct in activity_pcts.items():
        count = activity_counts[activity]
        print(f"  {activity:>10}: {count:>6,} points ({pct:>5.1f}%)")
    
    # Daily averages
    df['date'] = df.index.date
    daily_counts = df.groupby(['date', 'activity']).size().unstack(fill_value=0)
    daily_hours = daily_counts * 30 / 3600  # Convert 30-second windows to hours
    
    print(f"\nDaily Averages (hours):")
    for activity in daily_hours.columns:
        avg_hours = daily_hours[activity].mean()
        print(f"  {activity:>10}: {avg_hours:>5.1f} hours/day")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='Path to activity classification CSV file')
    parser.add_argument('--days', type=int, help='Number of days to plot (default: all)')
    parser.add_argument('--output_dir', help='Directory to save plots')
    args = parser.parse_args()
    
    # Load data
    df = load_activity_data(args.csv_file)
    
    # Extract participant ID from filename
    participant_id = Path(args.csv_file).stem
    
    # Print summary
    print_activity_summary(df, participant_id)
    
    # Set up save paths if output directory specified
    timeline_save = None
    daily_save = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timeline_save = output_dir / f"{participant_id}_timeline.png"
        daily_save = output_dir / f"{participant_id}_daily.png"
    
    # Create visualizations
    plot_activity_timeline(df, participant_id, days=args.days, save_path=timeline_save)
    plot_daily_activity_summary(df, participant_id, save_path=daily_save)

if __name__ == "__main__":
    main()