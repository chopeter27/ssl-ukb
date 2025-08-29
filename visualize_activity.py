"""
Visualize activity classification results as heatmaps

Usage:
    python visualize_activity_heatmap.py /home/peter.cho@vivosense.com/ssl-ukb/data/strath-step/OxWearables/actiNET/
    python visualize_activity_heatmap.py /path/to/actiNET/folder/ --individual --aggregate
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, time
import glob

def load_all_activity_data(folder_path):
    """Load all CSV files from the specified folder"""
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    
    participants_data = {}
    
    for csv_file in csv_files:
        participant_id = csv_file.stem
        try:
            # Read CSV with timestamp as index and parse dates
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Ensure we have the 'activity' column
            if 'activity' not in df.columns:
                print(f"Warning: No 'activity' column in {csv_file}, skipping...")
                continue
                
            # Handle any missing values by filling with 'missing'
            df['activity'] = df['activity'].fillna('missing')
            
            participants_data[participant_id] = df
            print(f"Loaded {participant_id}: {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return participants_data

def create_activity_matrix(df, participant_id):
    """Convert time series to day x time matrix"""
    
    if len(df) == 0:
        raise ValueError(f"No data for participant {participant_id}")
    
    # Create full date range to identify missing periods (already handled in inference)
    # The inference script should have filled gaps with 'missing' 
    df_work = df.copy()
    
    # Add time components
    df_work['date'] = df_work.index.date
    df_work['hour_min'] = df_work.index.strftime('%H:%M:%S')
    
    # For 30-second windows, we want to show time in HH:MM:SS format
    # Create pivot table: rows=days, columns=time_of_day
    try:
        activity_matrix = df_work.pivot_table(
            index='date', 
            columns='hour_min', 
            values='activity', 
            aggfunc='first'  # Take first value if multiple in same time bin
        )
    except Exception as e:
        print(f"Error creating matrix for {participant_id}: {e}")
        print(f"Data shape: {df_work.shape}")
        print(f"Date range: {df_work.index.min()} to {df_work.index.max()}")
        print(f"Sample data:\n{df_work.head()}")
        raise
    
    return activity_matrix

def plot_individual_heatmap(activity_matrix, participant_id, save_path=None):
    """Create heatmap for individual participant"""
    
    # Define activity mapping to numbers for coloring
    activity_map = {
        'sleep': 0,
        'sedentary': 1, 
        'light': 2,
        'moderate-vigorous': 3,
        'missing': 4
    }
    
    # Convert activities to numbers
    matrix_numeric = activity_matrix.replace(activity_map)
    
    # Ensure all possible values are represented (0-4) for consistent color mapping
    # This guarantees the colormap shows all activities even if some are missing
    matrix_numeric = matrix_numeric.fillna(4)  # Fill any remaining NaN with 'missing' (4)
    
    # Create custom colormap
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#808080']  # sleep, sedentary, light, moderate, missing
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 8))
    
    # Create heatmap
    sns.heatmap(matrix_numeric, 
                cmap=cmap, 
                cbar_kws={'ticks': [0, 1, 2, 3, 4]},
                ax=ax,
                xticklabels=120,  # Show every 120th time label (every hour for 30s windows)
                yticklabels=True)
    
    # Customize colorbar labels - always show all activity types
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Sleep', 'Sedentary', 'Light', 'Moderate-Vigorous', 'Missing'])
    
    # Format axes
    ax.set_title(f'Activity Patterns - Participant {participant_id}', fontsize=16, pad=20)
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Date', fontsize=12)
    
    # Format x-axis to show hours (every 120 * 30sec = 1 hour)
    x_ticks = range(0, len(activity_matrix.columns), 120)  
    x_labels = []
    for i in x_ticks:
        if i < len(activity_matrix.columns):
            time_str = activity_matrix.columns[i]
            # Extract hour from HH:MM:SS
            if ':' in time_str:
                hour = time_str.split(':')[0]
                x_labels.append(f"{hour}:00")
            else:
                x_labels.append(time_str)
        else:
            x_labels.append('')
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    
    # Format y-axis dates
    y_ticks = range(0, len(activity_matrix.index), max(1, len(activity_matrix.index)//10))
    y_labels = [str(activity_matrix.index[i]) for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual heatmap saved: {save_path}")
    
    plt.show()

def create_aggregate_matrix(participants_data):
    """Create aggregate activity matrix with proper participant tracking"""
    
    all_matrices = []
    participant_ids = list(participants_data.keys())
    failed_participants = []
    
    # Collect matrices and time coverage info
    time_coverage = {}  # Track how many participants have data at each time point
    
    print(f"\n=== Processing {len(participant_ids)} participants for aggregate matrix ===")
    
    for participant_id in participant_ids:
        df = participants_data[participant_id]
        print(f"\nProcessing participant {participant_id}:")
        print(f"  Raw data shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Unique activities: {df['activity'].unique()}")
        
        try:
            matrix = create_activity_matrix(df, participant_id)
            print(f"  Matrix shape: {matrix.shape}")
            print(f"  Matrix date range: {matrix.index.min()} to {matrix.index.max()}")
            print(f"  Time points: {len(matrix.columns)} (first: {matrix.columns[0]}, last: {matrix.columns[-1]})")
            
            # Check if matrix has any data
            if matrix.empty:
                print(f"  WARNING: Empty matrix for participant {participant_id}")
                failed_participants.append(participant_id)
                continue
            
            all_matrices.append((participant_id, matrix))
            
            # Count coverage for each time point
            for time_point in matrix.columns:
                if time_point not in time_coverage:
                    time_coverage[time_point] = 0
                time_coverage[time_point] += 1
            
            print(f"  SUCCESS: Added participant {participant_id}")
            
        except Exception as e:
            print(f"  ERROR processing {participant_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_participants.append(participant_id)
    
    if failed_participants:
        print(f"\nFailed to process participants: {failed_participants}")
    
    if not all_matrices:
        raise ValueError("No valid participant data to aggregate")
    
    print(f"\nSuccessfully processed {len(all_matrices)} participants")
    
    # Use a more lenient coverage requirement if we have few participants
    total_participants = len(all_matrices)
    if total_participants <= 2:
        min_coverage = 1  # If only 1-2 participants, accept any time point with data
    else:
        min_coverage = max(1, total_participants // 2)  # Require at least half the participants
    
    # Only keep time points where sufficient participants have data
    valid_time_points = [
        time_point for time_point, count in time_coverage.items() 
        if count >= min_coverage
    ]
    valid_time_points = sorted(valid_time_points)
    
    print(f"Time coverage analysis:")
    print(f"  Total unique time points: {len(time_coverage)}")
    print(f"  Minimum coverage required: {min_coverage} participants")
    print(f"  Valid time points: {len(valid_time_points)}")
    
    if len(valid_time_points) == 0:
        print("WARNING: No valid time points found! Using all available time points.")
        # Fallback: use all time points from any participant
        all_time_points = set()
        for _, matrix in all_matrices:
            all_time_points.update(matrix.columns)
        valid_time_points = sorted(list(all_time_points))
    
    # Stack participant data using only valid time points
    all_rows = []
    participant_info = []  # Track which participant and day each row belongs to
    
    for participant_idx, (participant_id, matrix) in enumerate(all_matrices):
        print(f"\nStacking data for participant {participant_id} (P{participant_idx+1}):")
        
        # Only use columns that have good coverage
        matrix_filtered = matrix.reindex(columns=valid_time_points, fill_value='missing')
        print(f"  Filtered matrix shape: {matrix_filtered.shape}")
        print(f"  Days: {len(matrix_filtered.index)}")
        
        # Add all days from this participant
        rows_added = 0
        for day_idx, date in enumerate(matrix_filtered.index):
            all_rows.append(matrix_filtered.loc[date])
            participant_info.append({
                'participant_id': participant_id,
                'participant_idx': participant_idx,
                'date': date,
                'label': f"P{participant_idx+1}_{date}"
            })
            rows_added += 1
        
        print(f"  Added {rows_added} days from participant {participant_id}")
    
    # Create the aggregate matrix
    aggregate_matrix = pd.DataFrame(all_rows)
    aggregate_matrix.index = [info['label'] for info in participant_info]
    aggregate_matrix.columns = valid_time_points
    
    print(f"\nFinal aggregate matrix:")
    print(f"  Shape: {aggregate_matrix.shape}")
    print(f"  Rows (total days): {len(aggregate_matrix)}")
    print(f"  Columns (time points): {len(valid_time_points)}")
    print(f"  Participants represented: {len(set(info['participant_idx'] for info in participant_info))}")
    
    return aggregate_matrix, [pid for pid, _ in all_matrices], participant_info

def plot_aggregate_heatmap(aggregate_matrix, participant_ids, participant_info, save_path=None):
    """Create aggregate heatmap across all participants with participant numbers on y-axis"""
    
    print(f"\n=== Creating aggregate heatmap ===")
    print(f"Aggregate matrix shape: {aggregate_matrix.shape}")
    print(f"Participant IDs: {participant_ids}")
    print(f"Participant info entries: {len(participant_info)}")
    
    # Verify participant info matches the matrix
    for i, info in enumerate(participant_info[:5]):  # Show first 5
        print(f"Row {i}: {info}")
    
    activity_map = {
        'sleep': 0,
        'sedentary': 1,
        'light': 2, 
        'moderate-vigorous': 3,
        'missing': 4
    }
    
    # Convert to numeric
    matrix_numeric = aggregate_matrix.replace(activity_map)
    
    # Ensure all possible values are represented (0-4) for consistent color mapping
    matrix_numeric = matrix_numeric.fillna(4)  # Fill any remaining NaN with 'missing' (4)
    
    print(f"Matrix numeric shape: {matrix_numeric.shape}")
    print(f"Matrix numeric value range: {matrix_numeric.min().min()} to {matrix_numeric.max().max()}")
    
    # Create custom colormap
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#808080']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create figure - adjust height based on number of rows
    fig_height = max(8, len(aggregate_matrix) * 0.1, len(participant_ids) * 1.5)
    fig, ax = plt.subplots(figsize=(24, fig_height))
    
    # Create heatmap
    sns.heatmap(matrix_numeric,
                cmap=cmap,
                cbar_kws={'ticks': [0, 1, 2, 3, 4]},
                ax=ax,
                xticklabels=120,  # Show every 120th time label
                yticklabels=False)  # We'll set custom y-labels
    
    # Customize colorbar - always show all activity types
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Sleep', 'Sedentary', 'Light', 'Moderate-Vigorous', 'Missing'])
    
    # Format axes
    ax.set_title(f'Aggregate Activity Patterns - All Participants (n={len(participant_ids)})', 
                fontsize=16, pad=20)
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Participants', fontsize=12)
    
    # Format x-axis  
    x_ticks = range(0, len(aggregate_matrix.columns), 120)  # Every hour
    x_labels = []
    for i in x_ticks:
        if i < len(aggregate_matrix.columns):
            time_str = aggregate_matrix.columns[i]
            if ':' in time_str:
                hour = time_str.split(':')[0]
                x_labels.append(f"{hour}:00")
            else:
                x_labels.append(time_str)
        else:
            x_labels.append('')
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    
    # Calculate participant boundaries and centers more carefully
    participant_boundaries = []
    participant_centers = []
    participant_day_counts = []
    current_pos = 0
    
    print(f"\n=== Calculating participant positions ===")
    
    # Get unique participant indices in order they appear
    unique_participant_indices = []
    participant_index_to_id = {}
    for info in participant_info:
        if info['participant_idx'] not in unique_participant_indices:
            unique_participant_indices.append(info['participant_idx'])
            participant_index_to_id[info['participant_idx']] = info['participant_id']
    
    print(f"Unique participant indices: {unique_participant_indices}")
    print(f"Participant index to ID mapping: {participant_index_to_id}")
    
    for participant_idx in unique_participant_indices:
        # Count days for this participant
        participant_days = sum(1 for info in participant_info if info['participant_idx'] == participant_idx)
        participant_day_counts.append(participant_days)
        
        print(f"Participant {participant_idx} ({participant_index_to_id[participant_idx]}): {participant_days} days, rows {current_pos} to {current_pos + participant_days - 1}")
        
        # Mark the center position for this participant's label
        participant_centers.append(current_pos + participant_days / 2)
        
        current_pos += participant_days
        
        # Add boundary line (except after the last participant)
        if participant_idx != unique_participant_indices[-1]:
            participant_boundaries.append(current_pos)
    
    print(f"Participant centers: {participant_centers}")
    print(f"Participant boundaries: {participant_boundaries}")
    
    # Draw participant boundary lines
    for boundary in participant_boundaries:
        ax.axhline(y=boundary, color='white', linewidth=2, alpha=0.8)
        print(f"Drew boundary line at row {boundary}")
    
    # Set y-axis labels at participant centers
    y_labels = []
    for i, participant_idx in enumerate(unique_participant_indices):
        participant_id = participant_index_to_id[participant_idx]
        y_labels.append(f'P{i+1}\n({participant_id})')
    
    ax.set_yticks(participant_centers)
    ax.set_yticklabels(y_labels, rotation=0, ha='right', va='center', fontsize=10)
    
    # Add text annotations showing number of days for each participant
    for i, (center_pos, day_count) in enumerate(zip(participant_centers, participant_day_counts)):
        ax.text(-len(aggregate_matrix.columns) * 0.02, center_pos, 
               f'{day_count} days', 
               ha='right', va='center', fontsize=9, alpha=0.7,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Aggregate heatmap saved: {save_path}")
    
    plt.show()

def print_aggregate_summary(participants_data):
    """Print summary statistics across all participants"""
    
    total_points = 0
    activity_totals = {}
    
    print(f"\n=== Aggregate Summary Across {len(participants_data)} Participants ===")
    
    for participant_id, df in participants_data.items():
        total_points += len(df)
        
        # Count activities for this participant
        activity_counts = df['activity'].value_counts()
        print(f"\nParticipant {participant_id}:")
        print(f"  Total points: {len(df):,}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Activities: {dict(activity_counts)}")
        
        for activity, count in activity_counts.items():
            if activity not in activity_totals:
                activity_totals[activity] = 0
            activity_totals[activity] += count
    
    print(f"\nTotal data points across all participants: {total_points:,}")
    
    print(f"\nAggregate Activity Distribution:")
    for activity, count in sorted(activity_totals.items()):
        pct = count / total_points * 100
        print(f"  {activity:>10}: {count:>8,} points ({pct:>5.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', help='Path to folder containing activity CSV files')
    parser.add_argument('--individual', action='store_true', help='Generate individual heatmaps')
    parser.add_argument('--aggregate', action='store_true', help='Generate aggregate heatmap') 
    parser.add_argument('--output_dir', help='Directory to save plots')
    args = parser.parse_args()
    
    # Default to both if neither specified
    if not args.individual and not args.aggregate:
        args.individual = True
        args.aggregate = True
    
    # Load all participant data
    participants_data = load_all_activity_data(args.folder_path)
    
    if not participants_data:
        print("No valid participant data found!")
        return
    
    # Print aggregate summary
    print_aggregate_summary(participants_data)
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual heatmaps
    if args.individual:
        print(f"\nGenerating individual heatmaps for {len(participants_data)} participants...")
        for participant_id, df in participants_data.items():
            try:
                matrix = create_activity_matrix(df, participant_id)
                save_path = None
                if args.output_dir:
                    save_path = output_dir / f"{participant_id}_heatmap.png"
                plot_individual_heatmap(matrix, participant_id, save_path)
            except Exception as e:
                print(f"Error creating heatmap for {participant_id}: {e}")
    
    # Generate aggregate heatmap
    if args.aggregate:
        print(f"\nGenerating aggregate heatmap...")
        try:
            aggregate_matrix, participant_ids, participant_info = create_aggregate_matrix(participants_data)
            save_path = None
            if args.output_dir:
                save_path = output_dir / "aggregate_heatmap.png"
            plot_aggregate_heatmap(aggregate_matrix, participant_ids, participant_info, save_path)
        except Exception as e:
            print(f"Error creating aggregate heatmap: {e}")

if __name__ == "__main__":
    main()