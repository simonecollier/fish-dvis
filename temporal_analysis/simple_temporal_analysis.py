#!/usr/bin/env python3
"""
Simple temporal analysis for understanding motion importance in DVIS-DAQ
"""

import torch
import numpy as np
import json
import cv2
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
import gc

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class SimpleTemporalAnalyzer:
    """Simple temporal analysis focusing on motion patterns"""
    
    def __init__(self, config):
        self.config = config
        
        # Load dataset
        with open(config.dataset_json_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_video_frames(self, video_path, max_frames=None):
        """Load video frames from directory"""
        video_dir = Path(video_path)
        if not video_dir.exists():
            print(f"Video directory not found: {video_dir}")
            return None
        
        # Find all frame files
        frame_files = sorted([f for f in video_dir.glob("*.jpg")])
        if not frame_files:
            print(f"No frame files found in {video_dir}")
            return None
        
        # Limit frames if specified
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        frames = []
        for frame_file in frame_files:
            try:
                frame = cv2.imread(str(frame_file))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 640))
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                continue
        
        if not frames:
            print(f"No frames loaded from {video_dir}")
            return None
        
        return np.array(frames)
    
    def analyze_temporal_patterns(self, video_frames, video_id):
        """Analyze temporal patterns in the video"""
        print(f"Analyzing temporal patterns for video {video_id}...")
        
        num_frames = len(video_frames)
        if num_frames < 2:
            print(f"Not enough frames for temporal analysis: {num_frames}")
            return None
        
        # 1. Optical Flow Analysis (motion between frames)
        optical_flow_magnitudes = []
        for i in range(num_frames - 1):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            
            # Convert to uint8 for OpenCV
            frame1_uint8 = (frame1 * 255).astype(np.uint8)
            frame2_uint8 = (frame2 * 255).astype(np.uint8)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1_uint8, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2_uint8, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                pyr_scale=0.5, levels=3, winsize=9,
                iterations=3, poly_n=5, poly_sigma=1.1,
                flags=0
            )
            
            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            optical_flow_magnitudes.append(magnitude)
        
        optical_flow_magnitudes = np.array(optical_flow_magnitudes)
        
        # 2. Frame Difference Analysis (simple temporal gradients)
        frame_differences = []
        for i in range(num_frames - 1):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            
            # Calculate absolute difference
            diff = np.abs(frame2 - frame1)
            frame_differences.append(diff)
        
        frame_differences = np.array(frame_differences)
        
        # 3. Temporal Statistics
        temporal_stats = {
            'optical_flow_mean': float(np.mean(optical_flow_magnitudes)),
            'optical_flow_max': float(np.max(optical_flow_magnitudes)),
            'optical_flow_std': float(np.std(optical_flow_magnitudes)),
            'frame_diff_mean': float(np.mean(frame_differences)),
            'frame_diff_max': float(np.max(frame_differences)),
            'frame_diff_std': float(np.std(frame_differences)),
            'temporal_consistency': float(1.0 / (1.0 + np.std(optical_flow_magnitudes))),  # Higher = more consistent
            'motion_intensity': float(np.mean(optical_flow_magnitudes) * np.std(optical_flow_magnitudes))  # Motion strength
        }
        
        # 4. Temporal importance scores (based on motion patterns)
        # Frames with more motion are likely more important for classification
        motion_scores = []
        for i in range(num_frames):
            if i == 0:
                # First frame: use motion to next frame
                motion_score = np.mean(optical_flow_magnitudes[0]) if len(optical_flow_magnitudes) > 0 else 0
            elif i == num_frames - 1:
                # Last frame: use motion from previous frame
                motion_score = np.mean(optical_flow_magnitudes[-1]) if len(optical_flow_magnitudes) > 0 else 0
            else:
                # Middle frames: average motion from previous and next frames
                prev_motion = np.mean(optical_flow_magnitudes[i-1])
                next_motion = np.mean(optical_flow_magnitudes[i])
                motion_score = (prev_motion + next_motion) / 2
            
            motion_scores.append(motion_score)
        
        temporal_stats['frame_importance_scores'] = [float(score) for score in motion_scores]
        temporal_stats['most_important_frame'] = int(np.argmax(motion_scores))
        temporal_stats['least_important_frame'] = int(np.argmin(motion_scores))
        
        print(f"Temporal analysis complete. Most important frame: {temporal_stats['most_important_frame']}")
        
        return {
            'optical_flow_magnitudes': optical_flow_magnitudes,
            'frame_differences': frame_differences,
            'temporal_stats': temporal_stats
        }
    
    def analyze_video(self, video_info):
        """Analyze a single video"""
        video_id = video_info.get('id', 'unknown')
        print(f"\nAnalyzing video {video_id}...")
        
        # Get video path
        if 'file_names' in video_info and video_info['file_names']:
            video_path = Path(self.config.video_data_root) / Path(video_info['file_names'][0]).parent
        else:
            print(f"No file_names found for video {video_id}")
            return None
        
        # Load video frames
        video_frames = self.load_video_frames(
            str(video_path), 
            max_frames=self.config.max_frames_per_video
        )
        
        if video_frames is None:
            print(f"Failed to load frames for video {video_id}")
            return None
        
        print(f"Loaded {len(video_frames)} frames")
        
        # Analyze temporal patterns
        analysis_result = self.analyze_temporal_patterns(video_frames, video_id)
        
        if analysis_result is None:
            print(f"Failed to analyze temporal patterns for video {video_id}")
            return None
        
        # Save results
        result = {
            'video_id': video_id,
            'num_frames': len(video_frames),
            'temporal_stats': analysis_result['temporal_stats']
        }
        
        # Save detailed results to file
        output_file = self.output_dir / f"temporal_analysis_{video_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(analysis_result, f)
        
        print(f"Temporal analysis saved to {output_file}")
        return result
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting simple temporal analysis...")
        
        # Create category_id to category_name mapping
        category_map = {}
        for category in self.dataset.get('categories', []):
            category_map[category['id']] = category['name']
        
        print(f"Category mapping: {category_map}")
        
        # Filter videos by species
        target_species_ids = []
        for category_id, category_name in category_map.items():
            if category_name in self.config.target_species:
                target_species_ids.append(category_id)
        
        print(f"Target species IDs: {target_species_ids}")
        
        # Create video to species mapping
        video_species_map = {}
        for annotation in self.dataset.get('annotations', []):
            video_id = annotation.get('video_id')
            category_id = annotation.get('category_id')
            if category_id in target_species_ids:
                video_species_map[video_id] = category_id
        
        # Filter videos
        filtered_videos = []
        for video in self.dataset['videos']:
            video_id = video.get('id')
            if video_id in video_species_map:
                video['category_id'] = video_species_map[video_id]
                filtered_videos.append(video)
        
        print(f"Found {len(filtered_videos)} videos for target species")
        
        # Limit number of videos
        if self.config.max_videos_per_species:
            filtered_videos = filtered_videos[:self.config.max_videos_per_species]
            print(f"Limited to {len(filtered_videos)} videos")
        
        # Analyze videos
        results = []
        for i, video_info in enumerate(tqdm(filtered_videos, desc="Analyzing videos")):
            print(f"\nProcessing video {i+1}/{len(filtered_videos)}")
            result = self.analyze_video(video_info)
            if result:
                results.append(result)
            
            # Clear memory
            gc.collect()
        
        # Save summary results
        summary_file = self.output_dir / "temporal_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {summary_file}")
        print(f"Processed {len(results)} videos successfully")
        
        return results

def main():
    """Main function"""
    # Create configuration
    config = create_memory_efficient_config(gpu_memory_gb=8)
    config.max_videos_per_species = 3
    config.max_frames_per_video = 10
    
    # Create analyzer
    analyzer = SimpleTemporalAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        print("\nTemporal Analysis Results:")
        for result in results:
            stats = result['temporal_stats']
            print(f"Video {result['video_id']}:")
            print(f"  Motion intensity: {stats['motion_intensity']:.6f}")
            print(f"  Temporal consistency: {stats['temporal_consistency']:.6f}")
            print(f"  Most important frame: {stats['most_important_frame']}")
            print(f"  Optical flow mean: {stats['optical_flow_mean']:.6f}")

if __name__ == "__main__":
    main()
