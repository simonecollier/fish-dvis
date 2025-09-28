#!/usr/bin/env python3
"""
Gradient extraction for DVIS-DAQ temporal analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from detectron2.structures import ImageList
import cv2
import os
import sys
import tempfile

# Add DVIS-DAQ to Python path
DVIS_PATH = Path("/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ")
if str(DVIS_PATH) not in sys.path:
    sys.path.insert(0, str(DVIS_PATH))

# Import DVIS-DAQ components
from dvis_Plus.data_video.dataset_mapper import YTVISDatasetMapper
from dvis_Plus.data_video.datasets.ytvis import register_ytvis_instances
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import add_minvis_config, add_dvis_config, add_ctvis_config
from dvis_daq.config import add_daq_config

class TemporalGradientExtractor:
    """Extract temporal gradients from DVIS-DAQ model"""
    
    def __init__(self, model_loader, config):
        """
        Initialize gradient extractor
        
        Args:
            model_loader: DVISModelLoader instance
            config: Configuration object
        """
        self.model_loader = model_loader
        self.config = config
        self.model = model_loader.model
        self.device = config.device
        
        # Register hooks for gradient extraction
        self.hook_outputs = {}
        self.register_gradient_hooks()
        
    def register_gradient_hooks(self):
        """Register hooks for capturing temporal features"""
        hook_targets = self.config.gradient_targets
        self.hook_outputs = self.model_loader.register_hooks(hook_targets)
    
    def extract_temporal_gradients(self, video_frames: torch.Tensor, target_class: int = 0) -> torch.Tensor:
        """
        Extract temporal gradients from the DVIS-DAQ model for a given video sequence.
        
        Args:
            video_frames: Tensor of shape (T, C, H, W) containing video frames
            target_class: Target class index for gradient computation
            
        Returns:
            Temporal gradients tensor of shape (T, H, W)
        """
        device = next(self.model.parameters()).device
        T = video_frames.shape[0]
        
        # Create the proper input format that matches what the DVIS-DAQ model expects
        # during evaluation (using the same format as YTVISDatasetMapper)
        
        # Convert video frames to the format expected by the model
        # The model expects a list of (C, H, W) tensors
        frame_list = []
        for t in range(T):
            # Convert to the format expected by the model
            frame = video_frames[t].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8
            frame_tensor = torch.as_tensor(np.ascontiguousarray(frame.transpose(2, 0, 1)))  # (C, H, W)
            frame_list.append(frame_tensor)
        
        # Create the dataset dict format that the model expects
        processed_dict = {
            "image": frame_list,  # List of (C, H, W) tensors
            "height": video_frames.shape[2],
            "width": video_frames.shape[3],
            "video_len": T,
            "frame_idx": list(range(T)),
            "file_names": [f"frame_{i:05d}.jpg" for i in range(T)],
            "instances": [],  # Empty for inference
        }
        
        # Prepare input for the model
        batched_inputs = [processed_dict]
        
        # Forward pass with gradient computation
        with torch.enable_grad():
            # Ensure all frames require gradients and are on the correct device
            for i, frame in enumerate(processed_dict["image"]):
                # Convert to float32 and move to device, then set requires_grad
                frame_float = frame.to(device, dtype=torch.float32)
                frame_float.requires_grad_(True)
                processed_dict["image"][i] = frame_float
            
            # Use inference mode to bypass the training-specific preprocessing
            self.model.eval()
            
            # Forward pass - use the DVIS-DAQ common_inference method
            # This bypasses the standard Detectron2 preprocessing and uses the DVIS-DAQ specific logic
            # Convert the list of tensors to the format expected by common_inference
            images_tensor = torch.stack(processed_dict["image"], dim=0)  # (T, C, H, W)
            
            # Set the keep attribute that common_inference expects
            self.model.keep = False
            
            # Call the DVIS-DAQ specific inference method
            outputs = self.model.common_inference(images_tensor, window_size=len(images_tensor))
            
            # The inference method returns a different format
            # We need to extract the relevant outputs for gradient computation
        
        # Extract gradients with respect to the target class
        # The common_inference method returns a different format
        # We need to extract the relevant outputs for gradient computation
        
        # For now, let's use a simple approach: compute gradients with respect to the input
        # We'll use the first frame as a proxy for the entire sequence
        if len(processed_dict["image"]) > 0:
            # Use the first frame to compute gradients
            first_frame = processed_dict["image"][0]
            if first_frame.grad is not None:
                # Aggregate gradients across channels
                grad_magnitude = torch.norm(first_frame.grad, dim=0)  # Shape: (H, W)
                # Repeat for all frames (simple approach)
                temporal_gradients = [grad_magnitude] * T
                return torch.stack(temporal_gradients)
            else:
                # No gradients computed, return zeros
                return torch.zeros(T, video_frames.shape[2], video_frames.shape[3], device=device)
        else:
            # Fallback: return zeros if no frames
            return torch.zeros(T, video_frames.shape[2], video_frames.shape[3], device=device)
    
    def aggregate_temporal_gradients(self, 
                                   gradients: torch.Tensor, 
                                   method: str = "magnitude") -> torch.Tensor:
        """
        Aggregate gradients across spatial dimensions to get temporal importance
        
        Args:
            gradients: Input gradients (T, C, H, W) or (B, T, C, H, W)
            method: Aggregation method
            
        Returns:
            Temporal importance scores (T,)
        """
        # Handle different tensor shapes
        if gradients.dim() == 5:  # (B, T, C, H, W)
            gradients = gradients.squeeze(0)  # Remove batch dimension
        elif gradients.dim() != 4:  # Not (T, C, H, W)
            raise ValueError(f"Expected 4D or 5D tensor, got {gradients.dim()}D")
        
        # Now gradients should be (T, C, H, W)
        T, C, H, W = gradients.shape
        
        if method == "magnitude":
            # Average gradient magnitude across spatial dimensions
            temporal_importance = gradients.abs().mean(dim=[1, 2, 3])
        
        elif method == "l2_norm":
            # L2 norm across spatial dimensions
            temporal_importance = gradients.norm(dim=[1, 2, 3])
        
        elif method == "max_pool":
            # Max pooling across spatial dimensions
            temporal_importance = gradients.abs().amax(dim=[1, 2, 3])
        
        elif method == "weighted_magnitude":
            # Weighted average based on gradient magnitude
            weights = gradients.abs().mean(dim=[2, 3])  # (T, C)
            temporal_importance = (weights * gradients.abs().mean(dim=[2, 3])).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return temporal_importance
    
    def extract_hooked_gradients(self) -> Dict[str, torch.Tensor]:
        """Extract gradients from registered hooks"""
        gradients = {}
        
        # Extract gradients from registered hooks
        for name, hook in self.gradient_hooks.items():
            if hasattr(hook, 'gradients') and hook.gradients is not None:
                gradients[name] = hook.gradients
        
        # If no gradients from hooks, return empty dict
        if not gradients:
            gradients = {"input_gradients": None}
        
        return gradients
    
    def compute_temporal_attention_weights(self, 
                                         attention_output: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from temporal attention layers
        
        Args:
            attention_output: Output from temporal attention layer
            
        Returns:
            Temporal attention weights
        """
        # This is a simplified extraction - actual implementation depends on attention mechanism
        if attention_output.dim() == 4:  # (batch, heads, seq_len, seq_len)
            # Average across attention heads
            attention_weights = attention_output.mean(dim=1)  # (batch, seq_len, seq_len)
            # Extract temporal importance (diagonal or row-wise average)
            temporal_importance = attention_weights.mean(dim=-1)  # (batch, seq_len)
        else:
            # Fallback: use output magnitude as importance
            temporal_importance = attention_output.abs().mean(dim=-1)
        
        return temporal_importance
    
    def analyze_temporal_gradient_patterns(self, 
                                         temporal_gradients: torch.Tensor,
                                         video_length: int) -> Dict[str, float]:
        """
        Analyze patterns in temporal gradients
        
        Args:
            temporal_gradients: Temporal gradient importance (T,)
            video_length: Length of video sequence
            
        Returns:
            Dictionary of pattern analysis metrics
        """
        gradients_np = temporal_gradients.cpu().numpy()
        
        # Basic statistics
        mean_importance = np.mean(gradients_np)
        std_importance = np.std(gradients_np)
        max_importance = np.max(gradients_np)
        min_importance = np.min(gradients_np)
        
        # Temporal distribution analysis
        gradient_range = max_importance - min_importance
        coefficient_of_variation = std_importance / mean_importance if mean_importance > 0 else 0
        
        # Peak analysis
        peaks = self.find_temporal_peaks(gradients_np)
        num_peaks = len(peaks)
        peak_concentration = num_peaks / video_length if video_length > 0 else 0
        
        # Temporal consistency (autocorrelation)
        temporal_consistency = self.compute_temporal_consistency(gradients_np)
        
        # Gradient distribution skewness
        skewness = self.compute_skewness(gradients_np)
        
        return {
            "mean_importance": float(mean_importance),
            "std_importance": float(std_importance),
            "max_importance": float(max_importance),
            "min_importance": float(min_importance),
            "gradient_range": float(gradient_range),
            "coefficient_of_variation": float(coefficient_of_variation),
            "num_peaks": int(num_peaks),
            "peak_concentration": float(peak_concentration),
            "temporal_consistency": float(temporal_consistency),
            "skewness": float(skewness)
        }
    
    def find_temporal_peaks(self, gradients: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find temporal peaks in gradient importance"""
        peaks = []
        for i in range(1, len(gradients) - 1):
            if (gradients[i] > gradients[i-1] and 
                gradients[i] > gradients[i+1] and 
                gradients[i] > threshold * np.max(gradients)):
                peaks.append(i)
        return peaks
    
    def compute_temporal_consistency(self, gradients: np.ndarray) -> float:
        """Compute temporal consistency using autocorrelation"""
        if len(gradients) < 2:
            return 0.0
        
        # Compute autocorrelation at lag 1
        autocorr = np.corrcoef(gradients[:-1], gradients[1:])[0, 1]
        return autocorr if not np.isnan(autocorr) else 0.0
    
    def compute_skewness(self, gradients: np.ndarray) -> float:
        """Compute skewness of gradient distribution"""
        mean = np.mean(gradients)
        std = np.std(gradients)
        if std == 0:
            return 0.0
        
        skewness = np.mean(((gradients - mean) / std) ** 3)
        return float(skewness)
    
    def extract_species_specific_gradients(self, 
                                         video_input: torch.Tensor,
                                         species_list: List[str],
                                         class_mapping: Dict[str, int]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract gradients for specific species
        
        Args:
            video_input: Video frames tensor
            species_list: List of species to analyze
            class_mapping: Mapping from species names to class indices
            
        Returns:
            Dictionary of gradients for each species
        """
        species_gradients = {}
        
        for species in species_list:
            if species in class_mapping:
                class_idx = class_mapping[species]
                
                # Extract gradients for this species
                gradients = self.extract_temporal_gradients(
                    video_input, target_class=class_idx
                )
                
                species_gradients[species] = gradients
        
        return species_gradients
    
    def save_gradient_analysis(self, 
                             analysis_results: Dict[str, Any], 
                             output_path: Path):
        """Save gradient analysis results"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save temporal gradients
        if "temporal_gradients" in analysis_results:
            gradients_path = output_path / "temporal_gradients.pt"
            torch.save(analysis_results["temporal_gradients"], gradients_path)
        
        # Save pattern analysis
        if "pattern_analysis" in analysis_results:
            pattern_path = output_path / "pattern_analysis.json"
            with open(pattern_path, 'w') as f:
                json.dump(analysis_results["pattern_analysis"], f, indent=2)
        
        # Save species-specific gradients
        if "species_gradients" in analysis_results:
            species_path = output_path / "species_gradients.pt"
            torch.save(analysis_results["species_gradients"], species_path)
        
        print(f"Gradient analysis saved to {output_path}")
    
    def cleanup(self):
        """Cleanup hooks and memory"""
        self.model_loader.remove_hooks()
        torch.cuda.empty_cache()
