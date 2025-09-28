#!/usr/bin/env python3
"""
Memory-efficient configuration for DVIS-DAQ temporal motion analysis
Optimized for 31-frame windows with GPU memory constraints
"""

import os
import sys
from pathlib import Path
from config import TemporalAnalysisConfig
from typing import Dict

class MemoryEfficientConfig(TemporalAnalysisConfig):
    """Memory-efficient configuration for temporal analysis"""
    
    def __init__(self, model_dir: str = None, gpu_memory_gb: float = 24.0):
        """
        Initialize memory-efficient configuration
        
        Args:
            model_dir: Path to model directory
            gpu_memory_gb: Available GPU memory in GB
        """
        super().__init__(model_dir)
        
        # Memory optimization settings
        self.gpu_memory_gb = gpu_memory_gb
        self.gpu_memory_limit = int(gpu_memory_gb * 1024)  # Convert to MiB
        
        # Adjust settings based on available memory
        self._optimize_for_memory()
        
        # Memory management settings
        self.enable_gradient_checkpointing = True
        self.save_intermediate_results = True
        self.cache_optical_flow = True
        self.parallel_processing = False
        
        # Chunked processing settings
        self.chunk_size = self._calculate_optimal_chunk_size()
        self.max_frames_per_video = 15  # Use 15-frame windows for analysis (reduced for memory)
        self.max_videos_per_species = min(5, self._calculate_max_videos())
        
        # Batch processing settings
        self.batch_size = 1  # Always process one video at a time
        self.enable_memory_monitoring = True
        self.force_cpu_offload = self.gpu_memory_gb < 8.0  # Use CPU if < 8GB GPU
        
        # Performance settings
        self.max_gpu_memory_usage = 0.75  # Use max 75% of GPU memory
        self.memory_cleanup_frequency = 3  # Cleanup every N videos
        
        print(f"Memory-efficient config initialized for {self.gpu_memory_gb}GB GPU")
        print(f"Chunk size: {self.chunk_size}, Max frames per video: {self.max_frames_per_video}")
    
    def _optimize_for_memory(self):
        """Optimize settings based on available GPU memory"""
        if self.gpu_memory_gb >= 24:
            # High-end GPU (24GB+)
            self.image_size = 1024
            self.enable_mixed_precision = True
            self.gradient_accumulation_steps = 1
            
        elif self.gpu_memory_gb >= 16:
            # Mid-range GPU (16GB)
            self.image_size = 896
            self.enable_mixed_precision = True
            self.gradient_accumulation_steps = 2
            
        elif self.gpu_memory_gb >= 8:
            # Lower-end GPU (8GB)
            self.image_size = 640
            self.enable_mixed_precision = True
            self.gradient_accumulation_steps = 4
            
        else:
            # Very limited GPU memory (< 8GB)
            self.image_size = 512
            self.enable_mixed_precision = False
            self.gradient_accumulation_steps = 8
            self.force_cpu_offload = True
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on GPU memory"""
        if self.gpu_memory_gb >= 24:
            return 8  # Process 8 frames at a time
        elif self.gpu_memory_gb >= 16:
            return 6  # Process 6 frames at a time
        elif self.gpu_memory_gb >= 8:
            return 4  # Process 4 frames at a time
        else:
            return 2  # Process 2 frames at a time
    
    def _calculate_max_frames(self) -> int:
        """Calculate maximum frames per video based on memory"""
        if self.gpu_memory_gb >= 24:
            return 31  # Full 31-frame analysis
        elif self.gpu_memory_gb >= 16:
            return 25  # Reduced but still substantial
        elif self.gpu_memory_gb >= 8:
            return 20  # Moderate reduction
        else:
            return 15  # Significant reduction
    
    def _calculate_max_videos(self) -> int:
        """Calculate maximum videos per species based on memory"""
        if self.gpu_memory_gb >= 24:
            return 10
        elif self.gpu_memory_gb >= 16:
            return 8
        elif self.gpu_memory_gb >= 8:
            return 5
        else:
            return 3
    
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """Estimate memory usage for analysis"""
        # Rough estimates based on model size and processing
        base_model_memory = 4.0  # GB for model weights
        per_frame_memory = 0.1   # GB per frame
        per_chunk_memory = self.chunk_size * per_frame_memory
        
        estimated_usage = {
            'model_memory': base_model_memory,
            'per_frame_memory': per_frame_memory,
            'per_chunk_memory': per_chunk_memory,
            'total_estimated': base_model_memory + per_chunk_memory,
            'available_memory': self.gpu_memory_gb,
            'memory_utilization': (base_model_memory + per_chunk_memory) / self.gpu_memory_gb
        }
        
        return estimated_usage
    
    def validate_memory_constraints(self) -> bool:
        """Validate that configuration fits within memory constraints"""
        usage_estimate = self.get_memory_usage_estimate()
        
        if usage_estimate['memory_utilization'] > 0.9:
            print(f"WARNING: Estimated memory usage ({usage_estimate['memory_utilization']:.1%}) exceeds 90%")
            print("Consider reducing chunk_size or max_frames_per_video")
            return False
        
        return True
    
    def to_dict(self):
        """Convert config to dictionary with memory settings"""
        base_dict = super().to_dict()
        
        # Add memory-specific settings
        memory_dict = {
            'gpu_memory_gb': self.gpu_memory_gb,
            'chunk_size': self.chunk_size,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'save_intermediate_results': self.save_intermediate_results,
            'force_cpu_offload': self.force_cpu_offload,
            'max_gpu_memory_usage': self.max_gpu_memory_usage,
            'memory_cleanup_frequency': self.memory_cleanup_frequency,
            'memory_usage_estimate': self.get_memory_usage_estimate()
        }
        
        base_dict.update(memory_dict)
        return base_dict


def create_memory_efficient_config(gpu_memory_gb: float = None) -> MemoryEfficientConfig:
    """
    Create memory-efficient configuration based on available GPU memory
    
    Args:
        gpu_memory_gb: Available GPU memory in GB. If None, will auto-detect.
        
    Returns:
        MemoryEfficientConfig instance
    """
    if gpu_memory_gb is None:
        # Auto-detect GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"Auto-detected GPU memory: {gpu_memory_gb:.1f}GB")
            else:
                gpu_memory_gb = 8.0  # Default assumption
                print("CUDA not available, using default 8GB assumption")
        except Exception as e:
            print(f"Error detecting GPU memory: {e}")
            gpu_memory_gb = 8.0  # Fallback
    
    config = MemoryEfficientConfig(gpu_memory_gb=gpu_memory_gb)
    
    # Validate configuration
    if not config.validate_memory_constraints():
        print("Memory constraints validation failed. Consider adjusting settings.")
    
    return config


# Example usage and testing
if __name__ == "__main__":
    # Test different memory configurations
    test_configs = [8, 16, 24]
    
    for memory_gb in test_configs:
        print(f"\n=== Testing {memory_gb}GB GPU Configuration ===")
        config = create_memory_efficient_config(memory_gb)
        
        print(f"Chunk size: {config.chunk_size}")
        print(f"Max frames per video: {config.max_frames_per_video}")
        print(f"Max videos per species: {config.max_videos_per_species}")
        print(f"Force CPU offload: {config.force_cpu_offload}")
        
        usage = config.get_memory_usage_estimate()
        print(f"Estimated memory utilization: {usage['memory_utilization']:.1%}")
        print(f"Validation passed: {config.validate_memory_constraints()}")
