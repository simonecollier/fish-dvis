#!/usr/bin/env python3
"""
Run gradient analysis on all videos in the dataset
"""

import os
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from patched_gradient_analyzer import PatchedGradientAnalyzer

def main():
    """Run gradient analysis on all videos"""
    
    # Create configuration
    config = create_memory_efficient_config()
    
    # Load dataset
    with open(config.dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Create video species mapping
    video_species_map = {}
    for ann in dataset['annotations']:
        video_id = ann['video_id']
        category_id = ann['category_id']
        if video_id not in video_species_map:
            video_species_map[video_id] = category_id
    
    # Get category names
    category_names = {cat['id']: cat['name'] for cat in dataset['categories']}
    
    # Initialize analyzer
    print("Initializing gradient analyzer...")
    analyzer = PatchedGradientAnalyzer(config)
    
    # Process all videos
    total_videos = len(dataset['videos'])
    print(f"Processing {total_videos} videos...")
    
    for i, video_info in enumerate(tqdm(dataset['videos'], desc="Processing videos")):
        video_id = video_info['id']
        video_name = video_info['file_names'][0].split('/')[0]  # Extract video name from path
        category_id = video_species_map.get(video_id, -1)
        category_name = category_names.get(category_id, 'unknown')
        
        print(f"\n🎬 Processing video {i+1}/{total_videos}: {video_name} ({category_name})")
        
        # Construct video path
        video_path = os.path.join(config.video_data_root, video_name)
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video directory not found: {video_path}")
            continue
        
        try:
            # Analyze video
            results = analyzer.analyze_video(video_info)
            
            if results:
                print(f"✅ Successfully analyzed {video_name}")
                print(f"   Mean gradient: {results.get('mean_gradient', 'N/A'):.6f}")
                print(f"   Max gradient: {results.get('max_gradient', 'N/A'):.6f}")
            else:
                print(f"❌ Failed to analyze {video_name}")
            
        except Exception as e:
            print(f"❌ Error processing video {video_name}: {e}")
            continue
        
        # Clear memory periodically
        if (i + 1) % 10 == 0:
            import gc
            gc.collect()
            print(f"🧹 Memory cleanup after {i+1} videos")
    
    print(f"\n🎉 Completed analysis of {total_videos} videos!")

if __name__ == "__main__":
    main()
