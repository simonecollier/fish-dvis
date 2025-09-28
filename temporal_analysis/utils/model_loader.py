#!/usr/bin/env python3
"""
Model loader for DVIS-DAQ temporal analysis
"""

import sys
import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Set up environment to match training configuration
os.environ['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'
os.environ['DETECTRON2_DATASETS'] = '/data'

# Add DVIS-DAQ to path
DVIS_PATH = Path("/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ")
sys.path.insert(0, str(DVIS_PATH))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Import DVIS-DAQ modules to register the model architecture
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import add_minvis_config, add_dvis_config, add_ctvis_config
from dvis_daq.config import add_daq_config

class DVISModelLoader:
    """Loader for DVIS-DAQ models"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize model loader
        
        Args:
            config_path: Path to config.yaml
            checkpoint_path: Path to model checkpoint
            device: Device to load model on
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        # Validate paths
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model = None
        self.cfg = None
        self.hooks = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load and parse configuration"""
        # Load YAML config
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create Detectron2 config with DVIS-DAQ support
        self.cfg = get_cfg()
        
        # Add DVIS-DAQ config functions
        add_deeplab_config(self.cfg)
        add_maskformer2_config(self.cfg)
        add_maskformer2_video_config(self.cfg)
        add_minvis_config(self.cfg)
        add_dvis_config(self.cfg)
        add_ctvis_config(self.cfg)
        add_daq_config(self.cfg)
        
        # Set basic config - handle missing keys gracefully
        try:
            self.cfg.merge_from_file(str(self.config_path))
        except KeyError as e:
            print(f"Warning: Some config keys may be missing: {e}")
            # Continue with basic config
        
        # Override device
        self.cfg.MODEL.DEVICE = self.device
        
        # Set model to eval mode
        if hasattr(self.cfg.MODEL, 'MASK_FORMER'):
            if hasattr(self.cfg.MODEL.MASK_FORMER, 'TEST'):
                self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
                self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
                self.cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        
        return config_dict
    
    def load_model(self) -> torch.nn.Module:
        """Load DVIS-DAQ model"""
        if self.cfg is None:
            self.load_config()
        
        # Build model
        self.model = build_model(self.cfg)
        
        # Load checkpoint
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(str(self.checkpoint_path))
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded DVIS-DAQ model from {self.checkpoint_path}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        return self.model
    
    def register_hooks(self, hook_targets: list) -> Dict[str, Any]:
        """
        Register hooks for gradient extraction
        
        Args:
            hook_targets: List of target layers to hook
            
        Returns:
            Dictionary of hook outputs
        """
        if self.model is None:
            self.load_model()
        
        hook_outputs = {}
        
        def create_hook(name):
            def hook(module, input, output):
                hook_outputs[name] = output
            return hook
        
        # Register hooks for temporal components
        for target in hook_targets:
            if target == "frame_embeds":
                # Hook frame embeddings from sem_seg_head
                if hasattr(self.model, 'sem_seg_head'):
                    self.hooks[target] = self.model.sem_seg_head.register_forward_hook(
                        create_hook(target)
                    )
            
            elif target == "track_queries":
                # Hook track queries from tracker
                if hasattr(self.model, 'tracker'):
                    self.hooks[target] = self.model.tracker.register_forward_hook(
                        create_hook(target)
                    )
            
            elif target == "temporal_attention":
                # Hook temporal attention layers
                if hasattr(self.model, 'tracker') and hasattr(self.model.tracker, 'transformer_time_self_attention_layers'):
                    # Hook the first temporal attention layer
                    self.hooks[target] = self.model.tracker.transformer_time_self_attention_layers[0].register_forward_hook(
                        create_hook(target)
                    )
            
            elif target == "backbone_features":
                # Hook backbone features
                if hasattr(self.model, 'backbone'):
                    self.hooks[target] = self.model.backbone.register_forward_hook(
                        create_hook(target)
                    )
            
            elif target == "mask_features":
                # Hook mask features
                if hasattr(self.model, 'sem_seg_head'):
                    self.hooks[target] = self.model.sem_seg_head.register_forward_hook(
                        create_hook(target)
                    )
        
        return hook_outputs
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        if self.model is None:
            self.load_model()
        
        info = {
            "model_type": type(self.model).__name__,
            "device": next(self.model.parameters()).device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        # Add temporal-specific info
        if hasattr(self.model, 'tracker'):
            info["has_tracker"] = True
            if hasattr(self.model.tracker, 'transformer_time_self_attention_layers'):
                info["temporal_attention_layers"] = len(self.model.tracker.transformer_time_self_attention_layers)
        
        if hasattr(self.model, 'refiner'):
            info["has_refiner"] = True
        
        if hasattr(self.model, 'sem_seg_head'):
            info["has_sem_seg_head"] = True
        
        return info
    
    def get_temporal_components(self) -> Dict[str, torch.nn.Module]:
        """Get temporal processing components"""
        if self.model is None:
            self.load_model()
        
        components = {}
        
        # Track module (main temporal processor)
        if hasattr(self.model, 'tracker'):
            components['tracker'] = self.model.tracker
            
            # Temporal attention layers
            if hasattr(self.model.tracker, 'transformer_time_self_attention_layers'):
                components['temporal_attention_layers'] = self.model.tracker.transformer_time_self_attention_layers
            
            # Cross attention layers
            if hasattr(self.model.tracker, 'transformer_cross_attention_layers'):
                components['cross_attention_layers'] = self.model.tracker.transformer_cross_attention_layers
        
        # Temporal refiner
        if hasattr(self.model, 'refiner'):
            components['refiner'] = self.model.refiner
        
        # Frame embeddings (from sem_seg_head)
        if hasattr(self.model, 'sem_seg_head'):
            components['sem_seg_head'] = self.model.sem_seg_head
        
        return components
    
    def prepare_input(self, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare input for DVIS-DAQ model
        
        Args:
            video_frames: Video frames tensor (T, C, H, W)
            
        Returns:
            Dictionary of model inputs
        """
        if self.cfg is None:
            self.load_config()
        
        # Normalize frames
        pixel_mean = torch.tensor(self.cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        pixel_std = torch.tensor(self.cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        
        normalized_frames = (video_frames - pixel_mean) / pixel_std
        
        # Prepare batched input
        batched_input = [{
            "image": normalized_frames,
            "height": video_frames.shape[-2],
            "width": video_frames.shape[-1]
        }]
        
        return batched_input
    
    def __del__(self):
        """Cleanup hooks on deletion"""
        self.remove_hooks()


def load_dvis_model(model_dir: str, device: str = "cuda") -> DVISModelLoader:
    """
    Convenience function to load DVIS model
    
    Args:
        model_dir: Directory containing model checkpoint and config
        device: Device to load model on
        
    Returns:
        DVISModelLoader instance
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_path = model_dir / "model_final.pth"
    
    return DVISModelLoader(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device
    )
