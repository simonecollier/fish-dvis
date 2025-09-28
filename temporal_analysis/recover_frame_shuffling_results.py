#!/usr/bin/env python3
"""
Recovery script for frame shuffling analysis results
Loads individual video .pkl files and recreates the summary JSON
"""

import pickle
import json
import numpy as np
from pathlib import Path
import glob
import sys

def recover_frame_shuffling_results(results_dir="temporal_analysis_results"):
    """
    Recover frame shuffling analysis results from individual .pkl files
    """
    results_path = Path(results_dir)
    
    # Find all frame shuffling analysis files
    pattern = "frame_shuffling_analysis_*.pkl"
    pkl_files = list(results_path.glob(pattern))
    
    print(f"Found {len(pkl_files)} frame shuffling analysis files:")
    for pkl_file in pkl_files:
        print(f"  - {pkl_file.name}")
    
    if not pkl_files:
        print("No frame shuffling analysis files found!")
        return None
    
    # Load and process each file
    summary_results = []
    
    for pkl_file in pkl_files:
        try:
            print(f"\nLoading {pkl_file.name}...")
            
            # Extract video ID from filename
            video_id = pkl_file.stem.replace("frame_shuffling_analysis_", "")
            
            # Load the pickle file
            with open(pkl_file, 'rb') as f:
                result_data = pickle.load(f)
            
            print(f"  Video ID: {video_id}")
            print(f"  Data keys: {list(result_data.keys())}")
            
            # Check if this is a multi-window result or single window result
            if 'aggregated_metrics' in result_data:
                # Multi-window aggregated result
                print(f"  Multi-window result with {result_data.get('num_windows', 'unknown')} windows")
                
                summary_result = {
                    'video_id': video_id,
                    'num_windows': result_data.get('num_windows', 1),
                    'mean_motion_reliance_ratio': float(result_data['aggregated_metrics']['mean_motion_reliance_ratio']),
                    'std_motion_reliance_ratio': float(result_data['aggregated_metrics']['std_motion_reliance_ratio']),
                    'mean_relative_drop': float(result_data['aggregated_metrics']['mean_relative_drop']),
                    'std_relative_drop': float(result_data['aggregated_metrics']['std_relative_drop']),
                    'mean_flow_correlation_drop': float(result_data['aggregated_metrics']['mean_flow_correlation_drop']),
                    'std_flow_correlation_drop': float(result_data['aggregated_metrics']['std_flow_correlation_drop']),
                    'mean_static_ratio': float(result_data['aggregated_metrics']['mean_static_ratio']) if not np.isnan(result_data['aggregated_metrics']['mean_static_ratio']) else None,
                    'std_static_ratio': float(result_data['aggregated_metrics']['std_static_ratio']) if not np.isnan(result_data['aggregated_metrics']['std_static_ratio']) else None,
                    'interpretation': result_data.get('interpretation', 'Unknown')
                }
                
                print(f"  Motion reliance ratio: {summary_result['mean_motion_reliance_ratio']:.3f} ± {summary_result['std_motion_reliance_ratio']:.3f}")
                print(f"  Relative drop: {summary_result['mean_relative_drop']:.3f} ± {summary_result['std_relative_drop']:.3f}")
                print(f"  Interpretation: {summary_result['interpretation']}")
                
            else:
                # Single window result (fallback)
                print(f"  Single window result")
                
                summary_result = {
                    'video_id': video_id,
                    'num_windows': 1,
                    'mean_motion_reliance_ratio': float(result_data['motion_reliance_ratio']),
                    'std_motion_reliance_ratio': 0.0,
                    'mean_relative_drop': float(result_data['relative_drop']),
                    'std_relative_drop': 0.0,
                    'mean_flow_correlation_drop': float(result_data['flow_correlation_drop']) if not np.isnan(result_data['flow_correlation_drop']) else None,
                    'std_flow_correlation_drop': 0.0,
                    'mean_static_ratio': float(result_data['static_ratio']) if not np.isnan(result_data['static_ratio']) else None,
                    'std_static_ratio': 0.0,
                    'interpretation': result_data.get('interpretation', 'Unknown')
                }
                
                print(f"  Motion reliance ratio: {summary_result['mean_motion_reliance_ratio']:.3f}")
                print(f"  Relative drop: {summary_result['mean_relative_drop']:.3f}")
                print(f"  Interpretation: {summary_result['interpretation']}")
            
            summary_results.append(summary_result)
            print(f"  ✅ Successfully processed {pkl_file.name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {pkl_file.name}: {e}")
            continue
    
    # Save the recovered summary
    if summary_results:
        summary_file = results_path / "frame_shuffling_summary_recovered.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print(f"\n✅ Successfully recovered {len(summary_results)} video results!")
        print(f"📁 Saved recovered summary to: {summary_file}")
        
        # Also update the original summary file
        original_summary_file = results_path / "frame_shuffling_summary.json"
        with open(original_summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print(f"📁 Updated original summary file: {original_summary_file}")
        
        # Print summary statistics
        print(f"\n📊 Recovery Summary:")
        print(f"  Total videos recovered: {len(summary_results)}")
        
        motion_ratios = [r['mean_motion_reliance_ratio'] for r in summary_results]
        print(f"  Mean motion reliance ratio: {np.mean(motion_ratios):.3f} ± {np.std(motion_ratios):.3f}")
        print(f"  Range: {np.min(motion_ratios):.3f} - {np.max(motion_ratios):.3f}")
        
        # Count by interpretation
        interpretations = [r['interpretation'] for r in summary_results]
        from collections import Counter
        interpretation_counts = Counter(interpretations)
        print(f"  Interpretations:")
        for interpretation, count in interpretation_counts.items():
            print(f"    - {interpretation}: {count} videos")
        
        return summary_results
    else:
        print("❌ No results were successfully recovered!")
        return None

def analyze_recovered_results(results_dir="temporal_analysis_results"):
    """
    Analyze the recovered results to identify interesting patterns
    """
    summary_file = Path(results_dir) / "frame_shuffling_summary_recovered.json"
    
    if not summary_file.exists():
        print("No recovered summary file found. Run recovery first.")
        return
    
    with open(summary_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n🔍 Analysis of Recovered Results ({len(results)} videos):")
    
    # Convert to numpy arrays for analysis
    motion_ratios = np.array([r['mean_motion_reliance_ratio'] for r in results])
    relative_drops = np.array([r['mean_relative_drop'] for r in results])
    flow_drops = np.array([r['mean_flow_correlation_drop'] for r in results if r['mean_flow_correlation_drop'] is not None])
    
    # Find videos with high motion reliance
    high_motion_videos = [r for r in results if r['mean_motion_reliance_ratio'] < 0.5]
    print(f"\n🎯 Videos with High Motion Reliance (ratio < 0.5): {len(high_motion_videos)}")
    for video in high_motion_videos[:10]:  # Show top 10
        print(f"  - Video {video['video_id']}: {video['mean_motion_reliance_ratio']:.3f} ({video['interpretation']})")
    
    # Find videos with high appearance reliance
    high_appearance_videos = [r for r in results if r['mean_motion_reliance_ratio'] > 0.9]
    print(f"\n👁️ Videos with High Appearance Reliance (ratio > 0.9): {len(high_appearance_videos)}")
    for video in high_appearance_videos[:10]:  # Show top 10
        print(f"  - Video {video['video_id']}: {video['mean_motion_reliance_ratio']:.3f} ({video['interpretation']})")
    
    # Statistical summary
    print(f"\n📈 Statistical Summary:")
    print(f"  Motion reliance ratio - Mean: {np.mean(motion_ratios):.3f}, Std: {np.std(motion_ratios):.3f}")
    print(f"  Relative drop - Mean: {np.mean(relative_drops):.3f}, Std: {np.std(relative_drops):.3f}")
    if len(flow_drops) > 0:
        print(f"  Flow correlation drop - Mean: {np.mean(flow_drops):.3f}, Std: {np.std(flow_drops):.3f}")
    
    # Distribution analysis
    print(f"\n📊 Distribution Analysis:")
    print(f"  Motion reliance ratios:")
    print(f"    < 0.3 (High motion): {np.sum(motion_ratios < 0.3)} videos")
    print(f"    0.3-0.5 (Moderate motion): {np.sum((motion_ratios >= 0.3) & (motion_ratios < 0.5))} videos")
    print(f"    0.5-0.7 (Mixed): {np.sum((motion_ratios >= 0.5) & (motion_ratios < 0.7))} videos")
    print(f"    0.7-0.9 (High appearance): {np.sum((motion_ratios >= 0.7) & (motion_ratios < 0.9))} videos")
    print(f"    > 0.9 (Very high appearance): {np.sum(motion_ratios > 0.9)} videos")

def main():
    """Main function"""
    print("🔄 Frame Shuffling Analysis Recovery Tool")
    print("=" * 50)
    
    # Recover the results
    recovered_results = recover_frame_shuffling_results()
    
    if recovered_results:
        # Analyze the recovered results
        analyze_recovered_results()
        
        print(f"\n✅ Recovery complete! You can now use the analyze_motion_reliance.py script")
        print(f"   to further analyze your recovered results.")
    else:
        print("❌ Recovery failed. Check the error messages above.")

if __name__ == "__main__":
    main()
