#!/usr/bin/env python3
"""
Main temporal analyzer for DVIS-DAQ motion awareness analysis
"""

import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import cv2
from PIL import Image

from config import TemporalAnalysisConfig
from utils.model_loader import DVISModelLoader
from gradient_extractor import TemporalGradientExtractor
from motion_correlator import MotionCorrelator

class DVIS_TemporalAnalyzer:
    """Main analyzer for DVIS-DAQ temporal motion awareness"""
    
    def __init__(self, config: TemporalAnalysisConfig = None):
        """
        Initialize temporal analyzer
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = TemporalAnalysisConfig()
        
        self.config = config
        self.config.validate()
        
        # Initialize components
        print("Loading DVIS-DAQ model...")
        self.model_loader = DVISModelLoader(
            config_path=str(self.config.config_path),
            checkpoint_path=str(self.config.model_path),
            device=self.config.device
        )
        
        # Actually load the model
        self.model = self.model_loader.load_model()
        
        self.gradient_extractor = TemporalGradientExtractor(self.model_loader, self.config)
        self.motion_correlator = MotionCorrelator(self.config)
        
        # Load dataset
        self.dataset = self.load_dataset()
        
        # Output paths
        self.output_paths = self.config.get_output_paths()
        
        print("Temporal analyzer initialized successfully!")
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load dataset for analysis"""
        # Load validation or training dataset
        if self.config.use_val_set:
            dataset_path = self.config.val_json
        else:
            dataset_path = self.config.train_json
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"Loaded dataset with {len(dataset['videos'])} videos")
        return dataset
    
    def load_video_frames(self, video_path: str, max_frames: int = None) -> torch.Tensor:
        """
        Load video frames from path
        
        Args:
            video_path: Path to video frames directory
            max_frames: Maximum number of frames to load
            
        Returns:
            Video frames tensor (T, C, H, W)
        """
        if max_frames is None:
            max_frames = self.config.max_frames_per_video
        
        # Load frames from directory
        frame_dir = Path(video_path)
        if not frame_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_path}")
        
        # Get frame files
        frame_files = sorted([f for f in frame_dir.glob("*.jpg")])
        if not frame_files:
            raise ValueError(f"No frame files found in {video_path}")
        
        # Limit frames
        frame_files = frame_files[:max_frames]
        
        # Load frames
        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert('RGB')
            frame_array = np.array(frame)
            frames.append(frame_array)
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(np.array(frames)).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        return frames_tensor
    
    def analyze_single_video(self, 
                           video_path: str,
                           species: str = None,
                           class_mapping: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Analyze temporal motion awareness for a single video
        
        Args:
            video_path: Path to video frames
            species: Species label (optional)
            class_mapping: Mapping from species to class indices
            
        Returns:
            Analysis results
        """
        print(f"Analyzing video: {video_path}")
        
        # Load video frames
        video_frames = self.load_video_frames(video_path)
        video_frames = video_frames.to(self.config.device)
        
        # Compute optical flow
        frames_np = video_frames.cpu().numpy().transpose(0, 2, 3, 1)  # (T, H, W, C)
        motion_magnitudes = self.motion_correlator.compute_optical_flow(frames_np)
        
        # Extract temporal gradients
        if species and class_mapping and species in class_mapping:
            target_class = class_mapping[species]
            temporal_gradients = self.gradient_extractor.extract_temporal_gradients(
                video_frames, target_class=target_class
            )
        else:
            temporal_gradients = self.gradient_extractor.extract_temporal_gradients(video_frames)
        
        # Ensure temporal gradients are on the correct device
        temporal_gradients = temporal_gradients.to(self.config.device)
        
        # Correlate motion with gradients
        correlation_results = self.motion_correlator.correlate_motion_gradients(
            motion_magnitudes, temporal_gradients
        )
        
        # Analyze temporal patterns
        temporal_patterns = self.motion_correlator.analyze_motion_gradient_temporal_patterns(
            motion_magnitudes, temporal_gradients
        )
        
        # Classify motion dependency
        classification = self.motion_correlator.classify_motion_dependency(correlation_results)
        
        # Analyze gradient patterns
        gradient_patterns = self.gradient_extractor.analyze_temporal_gradient_patterns(
            temporal_gradients, len(video_frames)
        )
        
        # Compute motion statistics
        motion_stats = self.motion_correlator.compute_motion_statistics(motion_magnitudes)
        
        return {
            "video_path": video_path,
            "species": species,
            "video_length": len(video_frames),
            "motion_magnitudes": motion_magnitudes,
            "temporal_gradients": temporal_gradients.cpu().numpy(),
            "correlation_results": correlation_results,
            "temporal_patterns": temporal_patterns,
            "classification": classification,
            "gradient_patterns": gradient_patterns,
            "motion_statistics": motion_stats,
            "gradients": gradients
        }
    
    def analyze_species_temporal_patterns(self, 
                                        species_list: List[str] = None,
                                        max_videos_per_species: int = None) -> Dict[str, Any]:
        """
        Analyze temporal patterns for specific species
        
        Args:
            species_list: List of species to analyze
            max_videos_per_species: Maximum videos per species
            
        Returns:
            Species-specific analysis results
        """
        if species_list is None:
            species_list = self.config.target_species
        
        if max_videos_per_species is None:
            max_videos_per_species = self.config.max_videos_per_species
        
        # Create class mapping (you'll need to adapt this to your dataset)
        class_mapping = self.create_class_mapping()
        
        species_results = {}
        
        for species in species_list:
            print(f"\nAnalyzing species: {species}")
            
            # Get videos for this species
            species_videos = self.get_videos_by_species(species)
            
            if len(species_videos) == 0:
                print(f"No videos found for species: {species}")
                continue
            
            # Limit videos
            species_videos = species_videos[:max_videos_per_species]
            
            # Analyze videos
            video_results = []
            for video_info in tqdm(species_videos, desc=f"Processing {species}"):
                try:
                    video_path = video_info["video_path"]
                    result = self.analyze_single_video(
                        video_path, species, class_mapping
                    )
                    video_results.append(result)
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")
                    continue
            
            if video_results:
                # Aggregate results for this species
                species_aggregated = self.aggregate_species_results(video_results)
                species_results[species] = species_aggregated
        
        return species_results
    
    def get_videos_by_species(self, species: str) -> List[Dict[str, Any]]:
        """Get videos for a specific species"""
        videos = []
        
        for video in self.dataset["videos"]:
            # Extract species from video annotations
            video_species = self.extract_video_species(video)
            
            if video_species == species:
                # Get video path
                video_path = self.get_video_path(video)
                if video_path:
                    videos.append({
                        "video_id": video["id"],
                        "video_path": video_path,
                        "species": species
                    })
        
        return videos
    
    def extract_video_species(self, video: Dict[str, Any]) -> str:
        """Extract species from video annotations"""
        # Get video ID
        video_id = video["id"]
        
        # Find annotations for this video
        if "annotations" in self.dataset:
            for annotation in self.dataset["annotations"]:
                if annotation["video_id"] == video_id:
                    # Get category ID and find corresponding species name
                    category_id = annotation["category_id"]
                    for category in self.dataset["categories"]:
                        if category["id"] == category_id:
                            return category["name"]
        
        return "Unknown"
    
    def get_video_path(self, video: Dict[str, Any]) -> str:
        """Get video path from video info"""
        # Extract video path from file_names
        if "file_names" in video and video["file_names"]:
            # Get directory from first frame
            first_frame = video["file_names"][0]
            video_dir = str(Path(first_frame).parent)
            
            # Construct full path using video_data_root
            full_path = Path(self.config.video_data_root) / video_dir
            
            # Check if path exists
            if full_path.exists():
                return str(full_path)
            else:
                print(f"Warning: Video path not found: {full_path}")
                return None
        
        return None
    
    def create_class_mapping(self) -> Dict[str, int]:
        """Create mapping from species names to class indices"""
        # Create mapping from the dataset categories
        class_mapping = {}
        
        if "categories" in self.dataset:
            for category in self.dataset["categories"]:
                class_mapping[category["name"]] = category["id"] - 1  # Convert to 0-based indexing
        
        print(f"Created class mapping: {class_mapping}")
        return class_mapping
    
    def aggregate_species_results(self, video_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across videos for a species"""
        if not video_results:
            return {}
        
        # Aggregate correlation results
        correlation_results = []
        classification_results = []
        motion_stats = []
        gradient_patterns = []
        
        for result in video_results:
            correlation_results.append(result["correlation_results"])
            classification_results.append(result["classification"])
            motion_stats.append(result["motion_statistics"])
            gradient_patterns.append(result["gradient_patterns"])
        
        # Compute averages
        avg_correlation = self.compute_average_correlation(correlation_results)
        avg_classification = self.compute_average_classification(classification_results)
        avg_motion_stats = self.compute_average_motion_stats(motion_stats)
        avg_gradient_patterns = self.compute_average_gradient_patterns(gradient_patterns)
        
        return {
            "num_videos": len(video_results),
            "average_correlation": avg_correlation,
            "average_classification": avg_classification,
            "average_motion_statistics": avg_motion_stats,
            "average_gradient_patterns": avg_gradient_patterns,
            "video_results": video_results
        }
    
    def compute_average_correlation(self, correlation_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average correlation across videos"""
        if not correlation_results:
            return {}
        
        avg_correlation = {}
        for key in correlation_results[0].keys():
            values = [result[key] for result in correlation_results if key in result]
            if values:
                avg_correlation[key] = float(np.mean(values))
        
        return avg_correlation
    
    def compute_average_classification(self, classification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average classification across videos"""
        if not classification_results:
            return {}
        
        # Count classifications
        classification_counts = {}
        confidence_scores = []
        
        for result in classification_results:
            classification = result["classification"]
            if classification not in classification_counts:
                classification_counts[classification] = 0
            classification_counts[classification] += 1
            
            confidence_scores.append(result["confidence"])
        
        # Find most common classification
        most_common = max(classification_counts.items(), key=lambda x: x[1])
        
        return {
            "most_common_classification": most_common[0],
            "classification_distribution": classification_counts,
            "average_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0
        }
    
    def compute_average_motion_stats(self, motion_stats: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average motion statistics across videos"""
        if not motion_stats:
            return {}
        
        avg_stats = {}
        for key in motion_stats[0].keys():
            values = [stat[key] for stat in motion_stats if key in stat]
            if values:
                avg_stats[key] = float(np.mean(values))
        
        return avg_stats
    
    def compute_average_gradient_patterns(self, gradient_patterns: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average gradient patterns across videos"""
        if not gradient_patterns:
            return {}
        
        avg_patterns = {}
        for key in gradient_patterns[0].keys():
            values = [pattern[key] for pattern in gradient_patterns if key in pattern]
            if values:
                avg_patterns[key] = float(np.mean(values))
        
        return avg_patterns
    
    def analyze_temporal_motion_awareness(self) -> Dict[str, Any]:
        """
        Run comprehensive temporal motion awareness analysis
        
        Returns:
            Complete analysis results
        """
        print("Starting comprehensive temporal motion awareness analysis...")
        
        # Analyze species-specific patterns
        species_results = self.analyze_species_temporal_patterns()
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(species_results)
        
        # Save results
        self.save_analysis_results(species_results, summary_stats)
        
        return {
            "species_results": species_results,
            "summary_statistics": summary_stats
        }
    
    def generate_summary_statistics(self, species_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all species"""
        summary = {
            "total_species_analyzed": len(species_results),
            "total_videos_analyzed": sum(result["num_videos"] for result in species_results.values()),
            "species_classifications": {},
            "overall_motion_dependency": {}
        }
        
        # Check if any species were successfully analyzed
        if len(species_results) == 0:
            summary["overall_motion_dependency"] = {
                "classification": "no_data",
                "motion_dependent_ratio": 0.0
            }
            return summary
        
        # Analyze classifications across species
        classifications = []
        for species, result in species_results.items():
            classification = result["average_classification"]["most_common_classification"]
            classifications.append(classification)
            summary["species_classifications"][species] = classification
        
        # Count classification types
        classification_counts = {}
        for classification in classifications:
            if classification not in classification_counts:
                classification_counts[classification] = 0
            classification_counts[classification] += 1
        
        summary["classification_distribution"] = classification_counts
        
        # Determine overall motion dependency
        motion_dependent_count = classification_counts.get("motion_dependent", 0)
        total_species = len(species_results)
        
        if total_species > 0:
            if motion_dependent_count / total_species > 0.5:
                overall_dependency = "motion_dependent"
            elif motion_dependent_count / total_species > 0.2:
                overall_dependency = "partially_motion_dependent"
            else:
                overall_dependency = "static_feature_dependent"
            
            summary["overall_motion_dependency"] = {
                "classification": overall_dependency,
                "motion_dependent_ratio": motion_dependent_count / total_species
            }
        else:
            summary["overall_motion_dependency"] = {
                "classification": "no_data",
                "motion_dependent_ratio": 0.0
            }
        
        return summary
    
    def save_analysis_results(self, 
                            species_results: Dict[str, Any],
                            summary_stats: Dict[str, Any]):
        """Save analysis results to files"""
        # Save species results
        species_path = self.output_paths["species_analysis"] / "species_results.json"
        with open(species_path, 'w') as f:
            json.dump(species_results, f, indent=2)
        
        # Save summary statistics
        summary_path = self.output_paths["reports"] / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate analysis report
        self.generate_analysis_report(species_results, summary_stats)
        
        print(f"Analysis results saved to {self.output_paths['base']}")
    
    def generate_analysis_report(self, 
                               species_results: Dict[str, Any],
                               summary_stats: Dict[str, Any]):
        """Generate comprehensive analysis report"""
        report_path = self.output_paths["reports"] / "temporal_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("DVIS-DAQ Temporal Motion Awareness Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total species analyzed: {summary_stats['total_species_analyzed']}\n")
            f.write(f"Total videos analyzed: {summary_stats['total_videos_analyzed']}\n")
            f.write(f"Overall motion dependency: {summary_stats['overall_motion_dependency']['classification']}\n")
            f.write(f"Motion dependent ratio: {summary_stats['overall_motion_dependency']['motion_dependent_ratio']:.2f}\n\n")
            
            f.write("SPECIES-SPECIFIC RESULTS\n")
            f.write("-" * 25 + "\n")
            
            for species, result in species_results.items():
                f.write(f"\n{species}:\n")
                f.write(f"  Videos analyzed: {result['num_videos']}\n")
                f.write(f"  Classification: {result['average_classification']['most_common_classification']}\n")
                f.write(f"  Average confidence: {result['average_classification']['average_confidence']:.3f}\n")
                
                # Motion statistics
                motion_stats = result['average_motion_statistics']
                f.write(f"  Average motion magnitude: {motion_stats.get('mean_motion', 0):.3f}\n")
                f.write(f"  Motion variance: {motion_stats.get('motion_variance', 0):.3f}\n")
                
                # Correlation results
                corr_results = result['average_correlation']
                f.write(f"  Pearson correlation: {corr_results.get('pearson_correlation', 0):.3f}\n")
                f.write(f"  Spearman correlation: {corr_results.get('spearman_correlation', 0):.3f}\n")
                f.write(f"  Mutual information: {corr_results.get('mutual_information', 0):.3f}\n")
        
        print(f"Analysis report generated: {report_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.gradient_extractor.cleanup()
        torch.cuda.empty_cache()


def main():
    """Example usage of the temporal analyzer"""
    # Initialize analyzer
    analyzer = DVIS_TemporalAnalyzer()
    
    try:
        # Run comprehensive analysis
        results = analyzer.analyze_temporal_motion_awareness()
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {analyzer.output_paths['base']}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
    
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
