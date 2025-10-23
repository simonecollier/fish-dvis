#!/usr/bin/env python3
"""
Integrated Attention Extraction and Visualization for DVIS-DAQ

This script extracts attention maps and creates visualizations that properly map
attention weights back to pixel coordinates in the original video frames.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add paths
sys.path.append('/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ')
sys.path.append('/home/simone/fish-dvis/attention_analysis')

from extract_attention_maps import AttentionExtractor, TargetedAttentionExtractor
from visualize_attention_maps import AttentionVisualizer
from combine_attention_heads import AttentionCombiner

# Set environment variables
os.environ['DETECTRON2_DATASETS'] = '/data'
os.environ['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedAttentionAnalyzer:
    """
    Integrated attention extraction and visualization
    """
    
    def __init__(self, model_path: str, output_dir: str = "/store/simone/attention_viz/"):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to the trained DVIS-DAQ model
            output_dir: Directory to save visualizations
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the targeted attention extractor (memory efficient)
        self.extractor = TargetedAttentionExtractor(model_path, str(self.output_dir))
        
    def extract_with_original_images(self, video_id: str, layer_idx: int, frame_idx: int) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract attention maps while preserving original and processed images
        
        Args:
            video_id: Video ID from the dataset
            layer_idx: Which backbone layer to extract
            frame_idx: Global frame index for visualization
            
        Returns:
            Tuple of (attention_maps, original_images, processed_images)
        """
        logger.info(f"Loading video {video_id} and extracting attention maps...")
        
        # Load video data using the same method as the extractor
        video_input = self.extractor._load_video_from_dataset(video_id)
        
        # Get original images (before preprocessing)
        # We need to load the raw images from file_names
        original_images = []
        if 'file_names' in video_input:
            from detectron2.data.detection_utils import read_image
            import os
            
            # Your dataset base path
            base_image_path = "/home/simone/shared-data/fishway_ytvis/all_videos"
            
            for file_name in video_input['file_names']:
                # Construct full path to the JPG file
                full_image_path = os.path.join(base_image_path, file_name)
                
                if os.path.exists(full_image_path):
                    # Load original image
                    orig_img = read_image(full_image_path, format='RGB')  # HWC format
                    orig_img_tensor = torch.from_numpy(orig_img).permute(2, 0, 1)  # CHW format
                    original_images.append(orig_img_tensor)
                    logger.info(f"Loaded original image: {full_image_path}, shape: {orig_img_tensor.shape}")
                else:
                    logger.warning(f"Original image not found: {full_image_path}")
                    # Create a dummy image of the same size as processed image
                    if len(original_images) > 0:
                        dummy_img = torch.zeros_like(original_images[0])
                    else:
                        # Fallback dummy image
                        dummy_img = torch.zeros(3, 480, 640)  # CHW format
                    original_images.append(dummy_img)
                    logger.warning(f"Using dummy image for missing file: {file_name}")
        else:
            logger.warning("No file_names found in video_input, cannot load original images")
            original_images = []
        
        # Get processed images (as fed to the model)
        processed_images = video_input['image']  # List of preprocessed tensors
        
        # Extract only the targeted attention map (memory efficient!)
        attention_maps = self.extractor.extract_for_visualization(
            video_id, 
            layer_idx, 
            frame_idx
        )
        
        # attention_maps is already returned from extract_for_visualization
        
        logger.info(f"Extracted {len(attention_maps)} attention maps")
        logger.info(f"Original images: {len(original_images)}")
        logger.info(f"Processed images: {len(processed_images)}")
        
        return attention_maps, original_images, processed_images
    
    def create_comprehensive_visualization(self, video_id: str, 
                                         layers_to_viz: List[int] = [0, 5, 10, 15, 20],
                                         heads_to_viz: List[int] = [0, 4, 8, 12],
                                         frames_to_viz: List[int] = [0, 5, 10]) -> None:
        """
        Create comprehensive attention visualizations
        
        Args:
            video_id: Video ID to analyze
            layers_to_viz: Which ViT layers to visualize
            heads_to_viz: Which attention heads to visualize  
            frames_to_viz: Which frames to visualize
        """
        # Extract targeted attention maps and images (memory efficient!)
        attention_maps, original_images, processed_images = self.extract_with_original_images(
            video_id, layer_idx, frame_idx
        )
        
        if not original_images:
            logger.error("Could not load original images, skipping visualization")
            return
            
        # Create visualizer
        visualizer = AttentionVisualizer(attention_maps, original_images, processed_images)
        
        # Create output directory for this video
        video_output_dir = self.output_dir / f"video_{video_id}"
        video_output_dir.mkdir(exist_ok=True)
        
        # 1. Create patch grid overlay
        logger.info("Creating patch grid overlay...")
        for frame_idx in frames_to_viz[:3]:  # Limit to first 3 frames
            if frame_idx < len(original_images):
                save_path = video_output_dir / f"patch_grid_frame_{frame_idx}.png"
                visualizer.create_patch_grid_overlay(frame_idx=frame_idx, save_path=str(save_path))
        
        # 2. Create attention visualizations for different layers and heads
        logger.info("Creating attention visualizations...")
        
        # Sample some interesting query patches (corners, center, etc.)
        total_patches = visualizer.total_spatial_patches
        query_patches = [
            1,  # Top-left patch
            visualizer.patches_w,  # Top-right patch  
            total_patches // 2,  # Center patch
            total_patches - visualizer.patches_w,  # Bottom-left patch
            total_patches  # Bottom-right patch
        ]
        
        for layer_idx in layers_to_viz:
            for head_idx in heads_to_viz:
                for frame_idx in frames_to_viz:
                    if frame_idx >= len(original_images):
                        continue
                        
                    for query_patch in query_patches[:2]:  # Limit to first 2 query patches
                        try:
                            save_path = video_output_dir / f"attention_L{layer_idx}_H{head_idx}_F{frame_idx}_Q{query_patch}.png"
                            visualizer.visualize_backbone_attention(
                                layer_idx=layer_idx,
                                head_idx=head_idx, 
                                query_patch=query_patch,
                                frame_idx=frame_idx,
                                save_path=str(save_path)
                            )
                        except Exception as e:
                            logger.warning(f"Could not create visualization for L{layer_idx}_H{head_idx}_F{frame_idx}_Q{query_patch}: {e}")
        
        # 3. Create attention statistics
        logger.info("Analyzing attention statistics...")
        stats = visualizer.analyze_attention_statistics()
        
        # Save statistics
        import json
        stats_path = video_output_dir / "attention_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Comprehensive visualization completed for video {video_id}")
        logger.info(f"Results saved to: {video_output_dir}")
    
    def quick_visualization(self, video_id: str, layer_idx: int = 10, head_idx: int = 0, 
                          frame_idx: int = 0, query_patch: int = None) -> None:
        """
        Create a quick visualization for exploration
        
        Args:
            video_id: Video ID to analyze
            layer_idx: Which ViT layer to visualize
            head_idx: Which attention head to visualize
            frame_idx: Which frame to visualize
            query_patch: Which patch to use as query (None = center patch)
        """
        # Extract targeted attention maps and images (memory efficient!)
        attention_maps, original_images, processed_images = self.extract_with_original_images(
            video_id, layer_idx, frame_idx
        )
        
        if not original_images:
            logger.error("Could not load original images, skipping visualization")
            return
            
        # Create visualizer
        visualizer = AttentionVisualizer(attention_maps, original_images, processed_images)
        
        # Use center patch if not specified
        if query_patch is None:
            query_patch = visualizer.total_spatial_patches // 2
        
        # Create output directory
        video_output_dir = self.output_dir / f"video_{video_id}_quick"
        video_output_dir.mkdir(exist_ok=True)
        
        # Create patch grid overlay
        patch_grid_path = video_output_dir / f"patch_grid_frame_{frame_idx}.png"
        visualizer.create_patch_grid_overlay(frame_idx=frame_idx, save_path=str(patch_grid_path))
        
        # Create attention visualization
        attention_path = video_output_dir / f"attention_L{layer_idx}_H{head_idx}_F{frame_idx}_Q{query_patch}.png"
        visualizer.visualize_backbone_attention(
            layer_idx=layer_idx,
            head_idx=head_idx,
            query_patch=query_patch,
            frame_idx=frame_idx,
            save_path=str(attention_path)
        )
        
        logger.info(f"Quick visualization completed!")
        logger.info(f"Patch grid: {patch_grid_path}")
        logger.info(f"Attention map: {attention_path}")
    
    def analyze_head_combinations(self, video_id: str, layer_idx: int = 10, frame_idx: int = 0, 
                                query_patch: int = None):
        """
        Analyze and visualize different head combination methods
        
        Args:
            video_id: Video ID from dataset
            layer_idx: Which backbone layer to analyze (0-23)
            frame_idx: Which frame to analyze
            query_patch: Which patch to use as query (None = center patch)
        """
        logger.info(f"Analyzing head combinations for video {video_id}")
        logger.info(f"Layer: {layer_idx}, Frame: {frame_idx}")
        
        # Extract attention maps and original images
        attention_maps, original_images, processed_images = self.extract_with_original_images(
            video_id, layer_idx, frame_idx
        )
        
        if not attention_maps:
            logger.error("No attention maps extracted, cannot analyze head combinations")
            return
        
        # Create combiner
        combiner = AttentionCombiner(attention_maps)
        
        # Use center patch if not specified
        if query_patch is None:
            # Estimate center patch (assuming square grid + CLS token)
            num_patches = None
            for key, tensor in attention_maps.items():
                if 'backbone_vit_layer' in key and tensor is not None:
                    num_patches = tensor.shape[-1]  # Last dimension is num_patches
                    break
            
            if num_patches:
                grid_size = int(np.sqrt(num_patches - 1))  # -1 for CLS token
                query_patch = 1 + (grid_size // 2) * grid_size + (grid_size // 2)  # Center patch
            else:
                query_patch = 216  # Fallback
        
        # Create output directory
        video_output_dir = self.output_dir / f"video_{video_id}_head_analysis"
        video_output_dir.mkdir(exist_ok=True)
        
        try:
            # Analyze head specialization
            logger.info("Analyzing head specialization...")
            head_stats = combiner.analyze_head_specialization(layer_idx, frame_idx)
            
            if head_stats:
                print(f"\n=== Head Specialization Analysis - Layer {layer_idx} ===")
                print(f"{'Head':<4} {'Local':<6} {'LongRange':<10} {'Sparsity':<8} {'Entropy':<8}")
                print("-" * 40)
                for head_idx, stats in head_stats.items():
                    print(f"{head_idx:2d}   "
                          f"{stats['local_focus']:.3f}  "
                          f"{stats['long_range_ratio']:.3f}      "
                          f"{stats['sparsity']:.3f}    "
                          f"{stats['attention_entropy']:.3f}")
            
            # Visualize head specialization
            specialization_plot_path = video_output_dir / f"head_specialization_L{layer_idx}_F{frame_idx}.png"
            combiner.visualize_head_specialization(layer_idx, str(specialization_plot_path))
            
            # Visualize combination methods
            combination_plot_path = video_output_dir / f"combination_methods_L{layer_idx}_F{frame_idx}_Q{query_patch}.png"
            combiner.visualize_combination_methods(layer_idx, query_patch, str(combination_plot_path))
            
            # Compare combination methods numerically
            combinations = combiner.compare_combination_methods(layer_idx, query_patch)
            if combinations:
                print(f"\n=== Attention Pattern Comparison - Query Patch {query_patch} ===")
                print(f"{'Method':<12} {'Max':<8} {'Mean':<8} {'Std':<8}")
                print("-" * 38)
                for method, pattern in combinations.items():
                    max_attn = float(pattern.max())
                    mean_attn = float(pattern.mean())
                    std_attn = float(pattern.std())
                    print(f"{method:<12} {max_attn:.4f}   {mean_attn:.4f}   {std_attn:.4f}")
            
            logger.info(f"Head combination analysis completed!")
            logger.info(f"Results saved to: {video_output_dir}")
            
        except Exception as e:
            logger.error(f"Error in head combination analysis: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and visualize DVIS-DAQ attention maps')
    parser.add_argument('--video-id', type=str, required=True, help='Video ID from the dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='/store/simone/attention_viz/', 
                       help='Output directory for visualizations')
    parser.add_argument('--quick', action='store_true', help='Create quick visualization only')
    parser.add_argument('--analyze-heads', action='store_true', help='Analyze attention head combinations')
    parser.add_argument('--layer', type=int, default=10, help='ViT layer to visualize')
    parser.add_argument('--head', type=int, default=0, help='Attention head to visualize (for quick mode)')
    parser.add_argument('--frame', type=int, default=0, help='Frame to visualize')
    parser.add_argument('--query-patch', type=int, default=None, help='Query patch index (default: center)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = IntegratedAttentionAnalyzer(args.model, args.output)
    
    if args.analyze_heads:
        # Head combination analysis
        analyzer.analyze_head_combinations(
            video_id=args.video_id,
            layer_idx=args.layer,
            frame_idx=args.frame,
            query_patch=args.query_patch
        )
    elif args.quick:
        # Quick visualization
        analyzer.quick_visualization(
            video_id=args.video_id,
            layer_idx=args.layer,
            head_idx=args.head,
            frame_idx=args.frame,
            query_patch=args.query_patch
        )
    else:
        # Comprehensive visualization
        analyzer.create_comprehensive_visualization(args.video_id)


if __name__ == "__main__":
    main()
