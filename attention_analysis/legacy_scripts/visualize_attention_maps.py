#!/usr/bin/env python3
"""
Attention Map Visualization for DVIS-DAQ

This script provides tools to visualize attention maps by properly mapping them back to pixel coordinates,
accounting for all preprocessing steps (resizing, cropping, padding, patch embedding).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """
    Visualizes attention maps by mapping them back to original pixel coordinates
    """
    
    def __init__(self, attention_maps: Dict[str, torch.Tensor], original_images: List[torch.Tensor], processed_images: List[torch.Tensor]):
        """
        Initialize the visualizer
        
        Args:
            attention_maps: Dictionary of attention maps from the model
            original_images: List of original video frames (before preprocessing)
            processed_images: List of preprocessed frames (as fed to model)
        """
        self.attention_maps = attention_maps
        self.original_images = original_images
        self.processed_images = processed_images
        
        # Store total video length for window calculations
        self.total_frames = len(original_images)
        
        # Determine preprocessing transformation parameters
        self._analyze_preprocessing()
        
    def _analyze_preprocessing(self):
        """
        Analyze the preprocessing pipeline to understand the transformation
        from original images to processed images
        """
        if len(self.original_images) == 0 or len(self.processed_images) == 0:
            logger.warning("No images provided for preprocessing analysis")
            return
            
        orig_h, orig_w = self.original_images[0].shape[-2:]
        proc_h, proc_w = self.processed_images[0].shape[-2:]
        
        logger.info(f"Original image size: {orig_h} x {orig_w}")
        logger.info(f"Processed image size: {proc_h} x {proc_w}")
        
        # Store transformation parameters
        self.orig_size = (orig_h, orig_w)
        self.proc_size = (proc_h, proc_w)
        self.scale_h = proc_h / orig_h
        self.scale_w = proc_w / orig_w
        
        # DINOv2 patch parameters - but we need to verify against actual attention maps
        self.patch_size = 14  # Standard DINOv2 patch size
        
        # Try to infer actual patch count from attention maps
        actual_patch_count = None
        for key, tensor in self.attention_maps.items():
            if 'backbone_vit' in key and len(tensor.shape) == 4:
                actual_patch_count = tensor.shape[-1]  # Last dimension should be patch count
                break
        
        if actual_patch_count is not None:
            # Work backwards from actual patch count
            self.total_patches = actual_patch_count
            self.total_spatial_patches = actual_patch_count - 1  # Subtract CLS token
            
            # Find best fit for patch grid
            import math
            sqrt_patches = math.sqrt(self.total_spatial_patches)
            
            # Try different factorizations
            possible_grids = []
            for h in range(1, int(sqrt_patches) + 10):
                if self.total_spatial_patches % h == 0:
                    w = self.total_spatial_patches // h
                    possible_grids.append((h, w))
            
            # Choose the grid closest to the expected aspect ratio
            target_ratio = proc_h / proc_w
            best_grid = min(possible_grids, key=lambda grid: abs(grid[0]/grid[1] - target_ratio))
            self.patches_h, self.patches_w = best_grid
            
            logger.info(f"Detected {actual_patch_count} total patches from attention maps")
            logger.info(f"Using patch grid: {self.patches_h} x {self.patches_w} = {self.total_spatial_patches} spatial patches + 1 CLS token")
        else:
            # Fallback to calculated values
            self.patches_h = proc_h // self.patch_size
            self.patches_w = proc_w // self.patch_size
            self.total_spatial_patches = self.patches_h * self.patches_w
            self.total_patches = self.total_spatial_patches + 1
            
            logger.warning(f"Could not detect patch count from attention maps, using calculated: {self.patches_h} x {self.patches_w} = {self.total_spatial_patches} spatial patches")
        
    def patch_idx_to_coordinates(self, patch_idx: int) -> Tuple[int, int, int, int]:
        """
        Convert patch index to pixel coordinates in the processed image
        
        Args:
            patch_idx: Patch index (0 = CLS token, 1+ = spatial patches)
            
        Returns:
            Tuple of (y1, x1, y2, x2) pixel coordinates in processed image
        """
        if patch_idx == 0:
            # CLS token - return center of image
            center_y, center_x = self.proc_size[0] // 2, self.proc_size[1] // 2
            return (center_y-7, center_x-7, center_y+7, center_x+7)
        
        # Convert to spatial patch index (subtract 1 for CLS token)
        spatial_idx = patch_idx - 1
        
        # Convert to 2D grid coordinates
        patch_row = spatial_idx // self.patches_w
        patch_col = spatial_idx % self.patches_w
        
        # Convert to pixel coordinates in processed image
        y1 = patch_row * self.patch_size
        x1 = patch_col * self.patch_size
        y2 = min(y1 + self.patch_size, self.proc_size[0])
        x2 = min(x1 + self.patch_size, self.proc_size[1])
        
        return (y1, x1, y2, x2)
    
    def patch_coords_to_original_image(self, y1: int, x1: int, y2: int, x2: int) -> Tuple[int, int, int, int]:
        """
        Convert patch coordinates from processed image to original image coordinates
        
        Args:
            y1, x1, y2, x2: Coordinates in processed image
            
        Returns:
            Tuple of (y1, x1, y2, x2) coordinates in original image
        """
        # Scale back to original image coordinates
        orig_y1 = int(y1 / self.scale_h)
        orig_x1 = int(x1 / self.scale_w)
        orig_y2 = int(y2 / self.scale_h)
        orig_x2 = int(x2 / self.scale_w)
        
        # Clamp to original image bounds
        orig_y1 = max(0, min(orig_y1, self.orig_size[0]))
        orig_x1 = max(0, min(orig_x1, self.orig_size[1]))
        orig_y2 = max(0, min(orig_y2, self.orig_size[0]))
        orig_x2 = max(0, min(orig_x2, self.orig_size[1]))
        
        return (orig_y1, orig_x1, orig_y2, orig_x2)
    
    def _find_window_for_frame(self, global_frame_idx: int):
        """
        Find which window contains the given global frame index
        
        Args:
            global_frame_idx: Frame index in the entire video (0 to total_frames-1)
            
        Returns:
            Tuple of (window_idx, local_frame_idx, window_size) or None if not found
        """
        if global_frame_idx < 0 or global_frame_idx >= self.total_frames:
            return None
        
        # Calculate which window this frame belongs to
        # Standard window size is 31, but the last window may be smaller
        window_size = 31  # Default window size
        window_idx = global_frame_idx // window_size
        local_frame_idx = global_frame_idx % window_size
        
        # Calculate the actual size of this window
        start_frame = window_idx * window_size
        end_frame = min((window_idx + 1) * window_size, self.total_frames)
        actual_window_size = end_frame - start_frame
        
        return (window_idx, local_frame_idx, actual_window_size)
    
    def visualize_backbone_attention(self, layer_idx: int = 0, head_idx: int = 0, query_patch: int = 1, 
                                   frame_idx: int = 0, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize backbone ViT attention for a specific query patch
        
        Args:
            layer_idx: Which ViT layer to visualize (0-23)
            head_idx: Which attention head to visualize (0-15) 
            query_patch: Which patch is the query (0 = CLS, 1+ = spatial patches)
            frame_idx: Global frame index in the video (0 to total_frames-1)
            save_path: Optional path to save the visualization
            
        Returns:
            Visualization as numpy array
        """
        # Find which window this frame belongs to
        window_info = self._find_window_for_frame(frame_idx)
        if window_info is None:
            raise ValueError(f"Could not find window information for frame {frame_idx}")
        
        window_idx, local_frame_idx, window_size = window_info
        logger.info(f"Global frame {frame_idx} maps to window {window_idx}, local frame {local_frame_idx} (window size: {window_size})")
        
        # Get the attention map from the correct window
        attn_key = f'window_{window_idx}_backbone_vit_layer_{layer_idx}'
        
        # Check if window-prefixed key exists, otherwise try without prefix (fallback)
        if attn_key not in self.attention_maps:
            fallback_key = f'backbone_vit_layer_{layer_idx}'
            if fallback_key in self.attention_maps:
                logger.warning(f"Window-specific attention map {attn_key} not found, using fallback {fallback_key}")
                attn_key = fallback_key
                # In this case, we need to check if the fallback has enough frames
                attention = self.attention_maps[attn_key]
                if local_frame_idx >= attention.shape[0]:
                    # Try using the available frames (likely from the last processed window)
                    logger.warning(f"Using last available frame {attention.shape[0]-1} instead of requested local frame {local_frame_idx}")
                    local_frame_idx = min(local_frame_idx, attention.shape[0]-1)
            else:
                available_keys = [k for k in self.attention_maps.keys() if 'backbone_vit' in k]
                raise ValueError(f"Neither {attn_key} nor {fallback_key} found. Available backbone keys: {available_keys[:5]}...")
        else:
            attention = self.attention_maps[attn_key]
        
        if local_frame_idx >= attention.shape[0]:
            raise ValueError(f"Local frame {local_frame_idx} not available in window {window_idx}, only {attention.shape[0]} frames")
        if head_idx >= attention.shape[1]:
            raise ValueError(f"Head {head_idx} not available, only {attention.shape[1]} heads")
            
        # Extract attention weights for this query patch
        # Handle both GPU and CPU tensors
        if attention.is_cuda:
            attn_weights = attention[local_frame_idx, head_idx, query_patch, :].cpu().numpy()  # [433]
        else:
            attn_weights = attention[local_frame_idx, head_idx, query_patch, :].numpy()  # [433]
        
        # Get original image (using global frame index)
        if frame_idx >= len(self.original_images):
            raise ValueError(f"Original image for frame {frame_idx} not available")
            
        original_img = self.original_images[frame_idx]
        if original_img.shape[0] == 3:  # CHW format
            original_img = original_img.permute(1, 2, 0)  # HWC format
        original_img = original_img.cpu().numpy()
        
        # Convert to uint8 if needed
        if original_img.dtype != np.uint8:
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            else:
                original_img = original_img.astype(np.uint8)
        
        # Create attention heatmap
        attention_map = np.zeros(self.orig_size)
        
        # Map each patch's attention weight to original image coordinates
        for patch_idx, weight in enumerate(attn_weights):
            if patch_idx == 0:  # Skip CLS token for spatial visualization
                continue
                
            # Get patch coordinates in processed image
            y1, x1, y2, x2 = self.patch_idx_to_coordinates(patch_idx)
            
            # Map to original image coordinates
            orig_y1, orig_x1, orig_y2, orig_x2 = self.patch_coords_to_original_image(y1, x1, y2, x2)
            
            # Add attention weight to the corresponding region
            attention_map[orig_y1:orig_y2, orig_x1:orig_x2] = weight
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Frame {frame_idx}')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(attention_map, cmap='hot', alpha=0.8)
        axes[1].set_title(f'Attention Heatmap\nLayer {layer_idx}, Head {head_idx}, Query Patch {query_patch}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(original_img)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.6)
        
        # Highlight query patch
        if query_patch > 0:  # Don't highlight CLS token
            y1, x1, y2, x2 = self.patch_idx_to_coordinates(query_patch)
            orig_y1, orig_x1, orig_y2, orig_x2 = self.patch_coords_to_original_image(y1, x1, y2, x2)
            rect = patches.Rectangle((orig_x1, orig_y1), orig_x2-orig_x1, orig_y2-orig_y1, 
                                   linewidth=3, edgecolor='lime', facecolor='none')
            axes[2].add_patch(rect)
            
        axes[2].set_title(f'Attention Overlay\n(Green box = Query Patch {query_patch})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        # Convert to numpy array for return
        fig.canvas.draw()
        viz_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        viz_array = viz_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return viz_array
    
    def create_patch_grid_overlay(self, frame_idx: int = 0, save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a visualization showing the patch grid overlay on the original image
        
        Args:
            frame_idx: Which frame to visualize
            save_path: Optional path to save the visualization
            
        Returns:
            Visualization as numpy array
        """
        # Get original image
        original_img = self.original_images[frame_idx]
        if original_img.shape[0] == 3:  # CHW format
            original_img = original_img.permute(1, 2, 0)  # HWC format
        original_img = original_img.cpu().numpy()
        
        # Convert to uint8 if needed
        if original_img.dtype != np.uint8:
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            else:
                original_img = original_img.astype(np.uint8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Frame {frame_idx}\nSize: {self.orig_size[0]}x{self.orig_size[1]}')
        axes[0].axis('off')
        
        # Image with patch grid overlay
        axes[1].imshow(original_img)
        
        # Draw patch boundaries
        for patch_idx in range(1, self.total_spatial_patches + 1):  # Skip CLS token
            y1, x1, y2, x2 = self.patch_idx_to_coordinates(patch_idx)
            orig_y1, orig_x1, orig_y2, orig_x2 = self.patch_coords_to_original_image(y1, x1, y2, x2)
            
            rect = patches.Rectangle((orig_x1, orig_y1), orig_x2-orig_x1, orig_y2-orig_y1, 
                                   linewidth=0.5, edgecolor='cyan', facecolor='none', alpha=0.7)
            axes[1].add_patch(rect)
            
            # Add patch number for first few patches
            if patch_idx <= 20:
                center_x = (orig_x1 + orig_x2) // 2
                center_y = (orig_y1 + orig_y2) // 2
                axes[1].text(center_x, center_y, str(patch_idx), 
                           ha='center', va='center', fontsize=8, color='yellow', weight='bold')
        
        axes[1].set_title(f'Patch Grid Overlay\n{self.patches_h}x{self.patches_w} = {self.total_spatial_patches} patches')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved patch grid visualization to {save_path}")
        
        # Convert to numpy array for return
        fig.canvas.draw()
        viz_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        viz_array = viz_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return viz_array
    
    def analyze_attention_statistics(self) -> Dict[str, any]:
        """
        Analyze attention statistics to understand what the model is focusing on
        
        Returns:
            Dictionary with attention statistics
        """
        stats = {}
        
        # Analyze backbone attention
        for layer_idx in range(24):  # 24 ViT layers
            attn_key = f'backbone_vit_layer_{layer_idx}'
            if attn_key not in self.attention_maps:
                continue
                
            attention = self.attention_maps[attn_key]  # [frames, heads, patches, patches]
            
            # Average attention across heads and frames
            avg_attention = attention.mean(dim=(0, 1))  # [patches, patches]
            
            # Analyze CLS token attention (what does CLS attend to?)
            cls_attention = avg_attention[0, 1:].cpu().numpy()  # Exclude self-attention
            
            # Find most attended patches
            top_patches = np.argsort(cls_attention)[-10:][::-1]  # Top 10 patches
            
            stats[f'layer_{layer_idx}'] = {
                'avg_attention_shape': avg_attention.shape,
                'cls_top_patches': top_patches.tolist(),
                'cls_attention_entropy': float(-np.sum(cls_attention * np.log(cls_attention + 1e-8))),
                'max_attention': float(cls_attention.max()),
                'min_attention': float(cls_attention.min())
            }
        
        return stats


def main():
    """
    Example usage of the attention visualizer
    """
    # This would be called after running attention extraction
    print("AttentionVisualizer ready for use!")
    print("\nExample usage:")
    print("1. Run attention extraction first:")
    print("   python run_attention_extraction.py --video-id 1 --model /path/to/model.pth")
    print("\n2. Load attention maps and images:")
    print("   # Load your attention maps, original images, and processed images")
    print("   visualizer = AttentionVisualizer(attention_maps, original_images, processed_images)")
    print("\n3. Visualize attention:")
    print("   visualizer.visualize_backbone_attention(layer_idx=0, head_idx=0, query_patch=100)")
    print("   visualizer.create_patch_grid_overlay(frame_idx=0)")


if __name__ == "__main__":
    main()
