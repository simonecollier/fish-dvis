#!/usr/bin/env python3
"""
Methods for combining attention heads and layers in vision transformers

This script demonstrates common techniques used in research for combining
attention information across heads and layers in ViT-based models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AttentionCombiner:
    """
    Utility class for combining attention maps across heads and layers
    """
    
    def __init__(self, attention_maps: Dict[str, torch.Tensor]):
        """
        Initialize with attention maps from extraction
        
        Args:
            attention_maps: Dictionary of attention maps from AttentionExtractor
        """
        self.attention_maps = attention_maps
        self.backbone_layers = self._extract_backbone_layers()
    
    def _extract_backbone_layers(self) -> Dict[int, torch.Tensor]:
        """Extract backbone attention maps organized by layer"""
        backbone_layers = {}
        
        for key, tensor in self.attention_maps.items():
            if 'backbone_vit_layer' in key:
                # Extract layer number from key like 'window_3_backbone_vit_layer_5'
                parts = key.split('_')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            backbone_layers[layer_idx] = tensor
                            break
                        except ValueError:
                            continue
        
        return backbone_layers
    
    def combine_heads_single_layer(self, layer_idx: int, method: str = 'mean') -> Optional[torch.Tensor]:
        """
        Combine attention heads for a single layer
        
        Args:
            layer_idx: Which layer to process
            method: Combination method ('mean', 'max', 'min', 'weighted_mean', 'top_k')
            
        Returns:
            Combined attention map: [batch_size, num_patches, num_patches]
        """
        if layer_idx not in self.backbone_layers:
            logger.warning(f"Layer {layer_idx} not found in backbone layers")
            return None
        
        attention = self.backbone_layers[layer_idx]
        
        # Expected shape: [batch_size, num_heads, num_patches, num_patches]
        if len(attention.shape) != 4:
            logger.warning(f"Unexpected attention shape for layer {layer_idx}: {attention.shape}")
            return None
        
        batch_size, num_heads, num_patches, _ = attention.shape
        
        if method == 'mean':
            # Simple average across heads
            combined = attention.mean(dim=1)  # [batch_size, num_patches, num_patches]
            
        elif method == 'max':
            # Take maximum attention across heads
            combined, _ = attention.max(dim=1)
            
        elif method == 'min':
            # Take minimum attention across heads
            combined, _ = attention.min(dim=1)
            
        elif method == 'weighted_mean':
            # Weight heads by their overall attention magnitude
            head_weights = attention.mean(dim=(2, 3))  # [batch_size, num_heads]
            head_weights = torch.softmax(head_weights, dim=1)  # Normalize
            
            # Apply weights
            weighted_attention = attention * head_weights.unsqueeze(-1).unsqueeze(-1)
            combined = weighted_attention.sum(dim=1)
            
        elif method == 'top_k':
            # Take top-k most attentive heads
            k = min(4, num_heads)  # Use top 4 heads or all if fewer
            head_importance = attention.mean(dim=(2, 3))  # [batch_size, num_heads]
            
            # Get top-k heads
            _, top_indices = torch.topk(head_importance, k, dim=1)
            
            # Select and average top-k heads
            selected_heads = []
            for b in range(batch_size):
                selected = attention[b, top_indices[b]]  # [k, num_patches, num_patches]
                selected_heads.append(selected.mean(dim=0))
            
            combined = torch.stack(selected_heads, dim=0)
            
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return combined
    
    def analyze_head_specialization(self, layer_idx: int, frame_idx: int = 0) -> Dict[int, Dict[str, float]]:
        """
        Analyze what each attention head specializes in
        
        Args:
            layer_idx: Which layer to analyze
            frame_idx: Which frame to analyze (for video data)
            
        Returns:
            Dictionary with head statistics
        """
        if layer_idx not in self.backbone_layers:
            logger.warning(f"Layer {layer_idx} not found")
            return {}
        
        attention = self.backbone_layers[layer_idx]
        
        if len(attention.shape) != 4:
            logger.warning(f"Unexpected attention shape: {attention.shape}")
            return {}
        
        # Use specific frame
        if frame_idx >= attention.shape[0]:
            frame_idx = 0
            
        frame_attention = attention[frame_idx]  # [num_heads, num_patches, num_patches]
        num_heads, num_patches, _ = frame_attention.shape
        
        head_stats = {}
        
        for head_idx in range(num_heads):
            head_attn = frame_attention[head_idx]  # [num_patches, num_patches]
            
            # Calculate various statistics
            stats = {
                'mean_attention': float(head_attn.mean()),
                'max_attention': float(head_attn.max()),
                'attention_entropy': self._calculate_entropy(head_attn),
                'local_focus': self._calculate_local_focus(head_attn),
                'long_range_ratio': self._calculate_long_range_ratio(head_attn),
                'sparsity': self._calculate_sparsity(head_attn),
            }
            
            head_stats[head_idx] = stats
        
        return head_stats
    
    def _calculate_entropy(self, attention_matrix: torch.Tensor) -> float:
        """Calculate attention entropy (higher = more distributed)"""
        # Flatten and normalize
        flat_attn = attention_matrix.flatten()
        flat_attn = flat_attn / (flat_attn.sum() + 1e-8)
        
        # Calculate entropy
        entropy = -(flat_attn * torch.log(flat_attn + 1e-8)).sum()
        return float(entropy)
    
    def _calculate_local_focus(self, attention_matrix: torch.Tensor, radius: int = 2) -> float:
        """Calculate how much attention focuses on local neighborhoods"""
        num_patches = attention_matrix.shape[0]
        
        # Assume square patch grid
        grid_size = int(np.sqrt(num_patches - 1))  # -1 for CLS token
        
        if grid_size * grid_size + 1 != num_patches:
            # Can't determine grid structure
            return 0.0
        
        local_attention = 0.0
        total_attention = 0.0
        
        # Iterate through spatial patches (skip CLS token at index 0)
        for i in range(1, num_patches):
            # Convert to 2D coordinates
            row = (i - 1) // grid_size
            col = (i - 1) % grid_size
            
            # Check attention to nearby patches
            for j in range(1, num_patches):
                target_row = (j - 1) // grid_size
                target_col = (j - 1) % grid_size
                
                # Calculate distance
                distance = abs(row - target_row) + abs(col - target_col)
                
                attention_weight = attention_matrix[i, j]
                total_attention += attention_weight
                
                if distance <= radius:
                    local_attention += attention_weight
        
        return float(local_attention / (total_attention + 1e-8))
    
    def _calculate_long_range_ratio(self, attention_matrix: torch.Tensor) -> float:
        """Calculate ratio of long-range to short-range attention"""
        num_patches = attention_matrix.shape[0]
        grid_size = int(np.sqrt(num_patches - 1))
        
        if grid_size * grid_size + 1 != num_patches:
            return 0.0
        
        short_range = 0.0
        long_range = 0.0
        
        for i in range(1, num_patches):
            row = (i - 1) // grid_size
            col = (i - 1) % grid_size
            
            for j in range(1, num_patches):
                target_row = (j - 1) // grid_size
                target_col = (j - 1) % grid_size
                
                distance = np.sqrt((row - target_row)**2 + (col - target_col)**2)
                attention_weight = attention_matrix[i, j]
                
                if distance <= 3:
                    short_range += attention_weight
                else:
                    long_range += attention_weight
        
        return float(long_range / (short_range + long_range + 1e-8))
    
    def _calculate_sparsity(self, attention_matrix: torch.Tensor) -> float:
        """Calculate attention sparsity (higher = more focused)"""
        # Gini coefficient as a measure of sparsity
        flat_attn = attention_matrix.flatten().sort()[0]
        n = len(flat_attn)
        
        cumsum = torch.cumsum(flat_attn, dim=0)
        gini = (2 * torch.arange(1, n + 1).float() - n - 1) * flat_attn
        gini = gini.sum() / (n * cumsum[-1] + 1e-8)
        
        return float(gini)
    
    def compare_combination_methods(self, layer_idx: int, query_patch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compare different head combination methods for a specific query patch
        
        Args:
            layer_idx: Which layer to analyze
            query_patch: Which patch to use as query (0 = CLS token)
            
        Returns:
            Dictionary with attention patterns for each method
        """
        methods = ['mean', 'max', 'min', 'weighted_mean', 'top_k']
        results = {}
        
        for method in methods:
            combined = self.combine_heads_single_layer(layer_idx, method)
            if combined is not None:
                # Extract attention pattern for the query patch
                attention_pattern = combined[0, query_patch, :]  # [num_patches]
                results[method] = attention_pattern
        
        return results
    
    def visualize_head_specialization(self, layer_idx: int, save_path: Optional[str] = None):
        """
        Create visualization of head specialization
        
        Args:
            layer_idx: Which layer to visualize
            save_path: Where to save the plot
        """
        head_stats = self.analyze_head_specialization(layer_idx)
        
        if not head_stats:
            print(f"No data available for layer {layer_idx}")
            return
        
        # Prepare data for plotting
        heads = list(head_stats.keys())
        metrics = ['mean_attention', 'attention_entropy', 'local_focus', 'long_range_ratio', 'sparsity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [head_stats[head][metric] for head in heads]
            
            axes[i].bar(heads, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Attention Head')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.suptitle(f'Head Specialization Analysis - Layer {layer_idx}', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Head specialization plot saved to: {save_path}")
        
        plt.show()
    
    def visualize_combination_methods(self, layer_idx: int, query_patch: int = 0, save_path: Optional[str] = None):
        """
        Visualize different head combination methods
        
        Args:
            layer_idx: Which layer to analyze
            query_patch: Which patch to use as query
            save_path: Where to save the plot
        """
        combinations = self.compare_combination_methods(layer_idx, query_patch)
        
        if not combinations:
            print(f"No data available for layer {layer_idx}")
            return
        
        num_patches = len(list(combinations.values())[0])
        # Calculate grid size more carefully
        spatial_patches = num_patches - 1  # -1 for CLS token
        grid_size = int(np.sqrt(spatial_patches))
        
        # Verify grid size is correct
        if grid_size * grid_size != spatial_patches:
            # Try to find the best rectangular grid
            factors = []
            for i in range(1, int(np.sqrt(spatial_patches)) + 1):
                if spatial_patches % i == 0:
                    factors.append((i, spatial_patches // i))
            
            if factors:
                # Choose the most square-like factor pair
                grid_h, grid_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            else:
                # Fallback: use approximate square
                grid_h = grid_w = grid_size
                logger.warning(f"Cannot create perfect grid for {spatial_patches} patches, using {grid_h}x{grid_w}")
        else:
            grid_h = grid_w = grid_size
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (method, attention_pattern) in enumerate(combinations.items()):
            if i >= len(axes):
                break
                
            # Convert to 2D grid for visualization (skip CLS token)
            spatial_attention = attention_pattern[1:].reshape(grid_h, grid_w)
            
            im = axes[i].imshow(spatial_attention.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            axes[i].set_title(f'{method.replace("_", " ").title()}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        # Remove empty subplot if exists
        if len(combinations) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        plt.suptitle(f'Head Combination Methods - Layer {layer_idx}, Query Patch {query_patch}', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combination methods plot saved to: {save_path}")
        
        plt.show()


def demonstrate_head_combination():
    """
    Demonstrate head combination techniques with example usage
    """
    print("=== Attention Head Combination Demo ===\n")
    
    # This would normally come from your AttentionExtractor
    print("1. Load attention maps from your extraction:")
    print("   from extract_attention_maps import TargetedAttentionExtractor")
    print("   extractor = TargetedAttentionExtractor(model_path, output_dir)")
    print("   attention_maps = extractor.extract_for_visualization(video_id, layer_idx, frame_idx)")
    print("   combiner = AttentionCombiner(attention_maps)")
    print()
    
    print("2. Combine heads for a single layer:")
    print("   # Simple average")
    print("   combined_mean = combiner.combine_heads_single_layer(layer_idx=5, method='mean')")
    print("   ")
    print("   # Weighted by head importance")
    print("   combined_weighted = combiner.combine_heads_single_layer(layer_idx=5, method='weighted_mean')")
    print("   ")
    print("   # Top-k most important heads")
    print("   combined_topk = combiner.combine_heads_single_layer(layer_idx=5, method='top_k')")
    print()
    
    print("3. Analyze head specialization:")
    print("   head_stats = combiner.analyze_head_specialization(layer_idx=5)")
    print("   for head_idx, stats in head_stats.items():")
    print("       print(f'Head {head_idx}: Local focus = {stats[\"local_focus\"]:.3f}')")
    print()
    
    print("4. Visualize combinations:")
    print("   combiner.visualize_head_specialization(layer_idx=5)")
    print("   combiner.visualize_combination_methods(layer_idx=5, query_patch=216)")
    print()
    
    print("=== Research-Based Recommendations ===\n")
    print("• **Mean combination**: Good general-purpose method, preserves all information")
    print("• **Weighted mean**: Emphasizes more important heads, often better for analysis")
    print("• **Top-k selection**: Focuses on most relevant heads, reduces noise")
    print("• **Individual analysis**: Essential for understanding model behavior")
    print()
    print("Most papers use **mean combination** for simplicity, but **weighted mean**")
    print("or **top-k** often provide more interpretable results.")


if __name__ == "__main__":
    demonstrate_head_combination()
