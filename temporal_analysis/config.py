#!/usr/bin/env python3
"""
Configuration for DVIS-DAQ temporal motion analysis
"""

import os
import sys
from pathlib import Path

# Set up environment to match training configuration
os.environ['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'
os.environ['DETECTRON2_DATASETS'] = '/data'

# Add DVIS-DAQ to Python path
DVIS_PATH = Path("/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ")
sys.path.insert(0, str(DVIS_PATH))

class TemporalAnalysisConfig:
    """Configuration class for temporal motion analysis"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize configuration
        
        Args:
            model_dir: Path to model directory containing checkpoints and config
        """
        # Default model directory
        if model_dir is None:
            model_dir = "/store/simone/dvis-model-outputs/trained_models/model3_unmasked"
        
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model_final.pth"
        self.config_path = self.model_dir / "config.yaml"
        
        # Data paths
        self.train_json = self.model_dir / "train.json"
        self.val_json = self.model_dir / "val.json"
        
        # Analysis settings
        self.use_val_set = True  # Use validation set for analysis
        self.max_videos_per_species = 10  # Limit videos per species for analysis (reduced for testing)
        self.max_frames_per_video = 10  # Maximum frames to analyze per video (reduced for memory)
        
        # GPU settings
        self.device = "cuda"
        self.gpu_memory_limit = 24564  # MiB (24GB)
        self.batch_size = 1  # Process one video at a time
        
        # Gradient extraction settings
        self.gradient_targets = [
            "frame_embeds",      # Frame embeddings
            "track_queries",     # Track queries
            "temporal_attention" # Temporal attention weights
        ]
        
        # Motion analysis settings
        self.optical_flow_method = "farneback"  # OpenCV optical flow method
        self.motion_correlation_threshold = 0.7  # High correlation threshold
        self.motion_low_correlation_threshold = 0.3  # Low correlation threshold
        
        # Temporal perturbation settings
        self.perturbation_types = ["shuffle", "drop", "reverse"]
        self.frame_drop_ratios = [0.5, 0.75]  # Drop 50%, 75% of frames
        
        # Analysis metrics
        self.metrics = [
            "motion_gradient_correlation",
            "temporal_perturbation_degradation", 
            "species_temporal_signature_distance"
        ]
        
        # Output settings
        self.output_dir = Path("temporal_analysis_results")
        self.save_intermediate_results = True
        self.save_visualizations = True
        
        # Performance settings
        self.enable_gradient_checkpointing = True
        self.cache_optical_flow = True
        self.parallel_processing = False  # Set to True if multiple GPUs available
        
        # Species to analyze (from your actual dataset)
        self.target_species = [
            "Chinook",
            "Coho", 
            "Atlantic",
            "Rainbow Trout",
            "Brown Trout"
        ]
        
        # Temporal analysis parameters
        self.temporal_window_size = 31  # From your config
        self.temporal_stride = 1
        self.min_temporal_length = 10  # Minimum frames for analysis
        
        # Gradient aggregation methods
        self.gradient_aggregation_methods = [
            "magnitude",  # Average gradient magnitude
            "l2_norm",    # L2 norm across spatial dimensions
            "max_pool"    # Max pooling across spatial dimensions
        ]
        
        # Statistical analysis settings
        self.confidence_level = 0.95
        self.min_sample_size = 3  # Minimum videos per species for analysis (reduced for testing)
        
        # Visualization settings
        self.plot_dpi = 300
        self.plot_format = "png"
        self.color_palette = "viridis"
        
        # Dataset configuration
        self.video_data_root = "/home/simone/shared-data/fishway_ytvis/all_videos"
        self.dataset_json_path = "/home/simone/shared-data/fishway_ytvis/val.json"  # Use the val.json from fishway_ytvis directory
        self.target_species = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
        
    def validate(self):
        """Validate configuration settings"""
        errors = []
        
        # Check model files exist
        if not self.model_path.exists():
            errors.append(f"Model checkpoint not found: {self.model_path}")
        
        if not self.config_path.exists():
            errors.append(f"Config file not found: {self.config_path}")
        
        # Check data files exist
        if self.use_val_set and not self.val_json.exists():
            errors.append(f"Validation JSON not found: {self.val_json}")
        
        if not self.use_val_set and not self.train_json.exists():
            errors.append(f"Training JSON not found: {self.train_json}")
        
        # Check GPU memory
        if self.gpu_memory_limit <= 0:
            errors.append("GPU memory limit must be positive")
        
        # Check analysis parameters
        if self.max_frames_per_video <= 0:
            errors.append("Max frames per video must be positive")
        
        if self.max_videos_per_species <= 0:
            errors.append("Max videos per species must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def get_output_paths(self):
        """Get output directory paths"""
        output_paths = {
            "base": self.output_dir,
            "temporal_heatmaps": self.output_dir / "temporal_heatmaps",
            "correlation_plots": self.output_dir / "correlation_plots", 
            "perturbation_results": self.output_dir / "perturbation_results",
            "species_analysis": self.output_dir / "species_analysis",
            "reports": self.output_dir / "reports",
            "cache": self.output_dir / "cache"
        }
        
        # Create directories
        for path in output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return output_paths
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "model_dir": str(self.model_dir),
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "use_val_set": self.use_val_set,
            "max_videos_per_species": self.max_videos_per_species,
            "max_frames_per_video": self.max_frames_per_video,
            "device": self.device,
            "gpu_memory_limit": self.gpu_memory_limit,
            "batch_size": self.batch_size,
            "gradient_targets": self.gradient_targets,
            "optical_flow_method": self.optical_flow_method,
            "motion_correlation_threshold": self.motion_correlation_threshold,
            "motion_low_correlation_threshold": self.motion_low_correlation_threshold,
            "perturbation_types": self.perturbation_types,
            "frame_drop_ratios": self.frame_drop_ratios,
            "metrics": self.metrics,
            "target_species": self.target_species,
            "temporal_window_size": self.temporal_window_size,
            "temporal_stride": self.temporal_stride,
            "min_temporal_length": self.min_temporal_length,
            "gradient_aggregation_methods": self.gradient_aggregation_methods,
            "confidence_level": self.confidence_level,
            "min_sample_size": self.min_sample_size
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# Default configuration
DEFAULT_CONFIG = TemporalAnalysisConfig()
