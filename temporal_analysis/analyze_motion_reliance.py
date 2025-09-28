#!/usr/bin/env python3
"""
Analyze Motion Reliance Results
Find videos with motion reliance (correlations below 1) from frame shuffling analysis
"""

import json
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_summary_results(summary_file):
    """Load the summary results JSON file"""
    with open(summary_file, 'r') as f:
        return json.load(f)

def load_detailed_results(pickle_file):
    """Load detailed results from pickle file"""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def analyze_motion_reliance(results_dir):
    """Analyze motion reliance across all videos"""
    results_dir = Path(results_dir)
    
    # Load summary results
    summary_file = results_dir / "frame_shuffling_summary.json"
    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return None
    
    summary_results = load_summary_results(summary_file)
    
    print(f"Found {len(summary_results)} videos in summary")
    
    # Create DataFrame for analysis
    data = []
    for result in summary_results:
        video_id = result['video_id']
        num_windows = result.get('num_windows', 1)
        
        if 'aggregated_metrics' in result:
            # Multi-window result
            metrics = result['aggregated_metrics']
            data.append({
                'video_id': video_id,
                'num_windows': num_windows,
                'mean_motion_reliance_ratio': metrics['mean_motion_reliance_ratio'],
                'std_motion_reliance_ratio': metrics['std_motion_reliance_ratio'],
                'mean_relative_drop': metrics['mean_relative_drop'],
                'mean_flow_correlation_drop': metrics['mean_flow_correlation_drop'],
                'interpretation': result['interpretation']
            })
        else:
            # Single window result
            data.append({
                'video_id': video_id,
                'num_windows': num_windows,
                'mean_motion_reliance_ratio': result.get('mean_motion_reliance_ratio', result.get('motion_reliance_ratio', np.nan)),
                'std_motion_reliance_ratio': result.get('std_motion_reliance_ratio', 0.0),
                'mean_relative_drop': result.get('mean_relative_drop', result.get('relative_drop', np.nan)),
                'mean_flow_correlation_drop': result.get('mean_flow_correlation_drop', result.get('flow_correlation_drop', np.nan)),
                'interpretation': result['interpretation']
            })
    
    df = pd.DataFrame(data)
    
    # Find videos with motion reliance
    print("\n=== MOTION RELIANCE ANALYSIS ===")
    
    # High motion reliance (ratio < 0.5)
    high_motion = df[df['mean_motion_reliance_ratio'] < 0.5]
    print(f"\nHigh Motion Reliance (ratio < 0.5): {len(high_motion)} videos")
    if len(high_motion) > 0:
        print("Videos with high motion reliance:")
        for _, row in high_motion.iterrows():
            print(f"  Video {row['video_id']}: ratio = {row['mean_motion_reliance_ratio']:.3f} ± {row['std_motion_reliance_ratio']:.3f}")
    
    # Moderate motion reliance (0.5 <= ratio < 0.7)
    moderate_motion = df[(df['mean_motion_reliance_ratio'] >= 0.5) & (df['mean_motion_reliance_ratio'] < 0.7)]
    print(f"\nModerate Motion Reliance (0.5 <= ratio < 0.7): {len(moderate_motion)} videos")
    if len(moderate_motion) > 0:
        print("Videos with moderate motion reliance:")
        for _, row in moderate_motion.iterrows():
            print(f"  Video {row['video_id']}: ratio = {row['mean_motion_reliance_ratio']:.3f} ± {row['std_motion_reliance_ratio']:.3f}")
    
    # Mixed reliance (0.7 <= ratio < 0.9)
    mixed_reliance = df[(df['mean_motion_reliance_ratio'] >= 0.7) & (df['mean_motion_reliance_ratio'] < 0.9)]
    print(f"\nMixed Reliance (0.7 <= ratio < 0.9): {len(mixed_reliance)} videos")
    
    # High appearance reliance (ratio >= 0.9)
    high_appearance = df[df['mean_motion_reliance_ratio'] >= 0.9]
    print(f"\nHigh Appearance Reliance (ratio >= 0.9): {len(high_appearance)} videos")
    
    # Overall statistics
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Mean motion reliance ratio: {df['mean_motion_reliance_ratio'].mean():.3f} ± {df['mean_motion_reliance_ratio'].std():.3f}")
    print(f"Median motion reliance ratio: {df['mean_motion_reliance_ratio'].median():.3f}")
    print(f"Range: {df['mean_motion_reliance_ratio'].min():.3f} - {df['mean_motion_reliance_ratio'].max():.3f}")
    
    # Find videos with strongest motion reliance
    print(f"\n=== TOP 10 VIDEOS WITH STRONGEST MOTION RELIANCE ===")
    top_motion = df.nsmallest(10, 'mean_motion_reliance_ratio')
    for _, row in top_motion.iterrows():
        print(f"Video {row['video_id']}: ratio = {row['mean_motion_reliance_ratio']:.3f} ± {row['std_motion_reliance_ratio']:.3f}")
    
    # Find videos with strongest appearance reliance
    print(f"\n=== TOP 10 VIDEOS WITH STRONGEST APPEARANCE RELIANCE ===")
    top_appearance = df.nlargest(10, 'mean_motion_reliance_ratio')
    for _, row in top_appearance.iterrows():
        print(f"Video {row['video_id']}: ratio = {row['mean_motion_reliance_ratio']:.3f} ± {row['std_motion_reliance_ratio']:.3f}")
    
    return df

def analyze_individual_windows(results_dir, video_id):
    """Analyze individual windows for a specific video"""
    results_dir = Path(results_dir)
    
    # Load detailed results
    pickle_file = results_dir / f"frame_shuffling_analysis_{video_id}.pkl"
    if not pickle_file.exists():
        print(f"Detailed results not found: {pickle_file}")
        return None
    
    detailed_results = load_detailed_results(pickle_file)
    
    print(f"\n=== DETAILED ANALYSIS FOR VIDEO {video_id} ===")
    print(f"Number of windows: {detailed_results['num_windows']}")
    
    # Analyze individual windows
    window_results = detailed_results['window_results']
    
    print(f"\nIndividual Window Analysis:")
    for i, window_result in enumerate(window_results):
        window_info = window_result['window_info']
        motion_ratio = window_result['motion_reliance_ratio']
        relative_drop = window_result['relative_drop']
        
        print(f"  Window {i+1} (frames {window_info['start_frame']}-{window_info['end_frame']}):")
        print(f"    Motion reliance ratio: {motion_ratio:.3f}")
        print(f"    Relative drop: {relative_drop:.3f}")
        
        # Check if this window shows motion reliance
        if motion_ratio < 0.7:
            print(f"    → SHOWS MOTION RELIANCE!")
        elif motion_ratio < 0.9:
            print(f"    → Mixed reliance")
        else:
            print(f"    → Appearance reliance")
    
    return detailed_results

def create_visualization(df, output_dir):
    """Create visualization of motion reliance results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['mean_motion_reliance_ratio'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(df['mean_motion_reliance_ratio'].median(), color='red', linestyle='--', label=f'Median: {df["mean_motion_reliance_ratio"].median():.3f}')
    plt.xlabel('Motion Reliance Ratio')
    plt.ylabel('Number of Videos')
    plt.title('Distribution of Motion Reliance Ratios')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['mean_motion_reliance_ratio'], df['mean_relative_drop'], alpha=0.6)
    plt.xlabel('Motion Reliance Ratio')
    plt.ylabel('Relative Drop')
    plt.title('Motion Reliance vs Relative Drop')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['mean_motion_reliance_ratio'], df['mean_flow_correlation_drop'], alpha=0.6)
    plt.xlabel('Motion Reliance Ratio')
    plt.ylabel('Flow Correlation Drop')
    plt.title('Motion Reliance vs Flow Correlation Drop')
    
    plt.subplot(2, 2, 4)
    # Categorize videos
    categories = []
    for ratio in df['mean_motion_reliance_ratio']:
        if ratio < 0.5:
            categories.append('High Motion')
        elif ratio < 0.7:
            categories.append('Moderate Motion')
        elif ratio < 0.9:
            categories.append('Mixed')
        else:
            categories.append('High Appearance')
    
    category_counts = pd.Series(categories).value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Reliance Types')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motion_reliance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_dir / 'motion_reliance_analysis.png'}")

def main():
    """Main function"""
    results_dir = "temporal_analysis_results"
    
    # Analyze all videos
    df = analyze_motion_reliance(results_dir)
    
    if df is not None:
        # Create visualization
        create_visualization(df, results_dir)
        
        # Save detailed analysis
        analysis_file = Path(results_dir) / "motion_reliance_analysis.csv"
        df.to_csv(analysis_file, index=False)
        print(f"\nDetailed analysis saved to: {analysis_file}")
        
        # Example: Analyze specific video in detail
        if len(df) > 0:
            # Find video with strongest motion reliance
            strongest_motion_video = df.loc[df['mean_motion_reliance_ratio'].idxmin(), 'video_id']
            print(f"\nAnalyzing video with strongest motion reliance: {strongest_motion_video}")
            analyze_individual_windows(results_dir, strongest_motion_video)

if __name__ == "__main__":
    main()
