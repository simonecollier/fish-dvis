import torch
import torch.nn as nn
import logging
import math

class RefinerAttentionHook:
    """Hook to capture attention weights from refiner MultiheadAttention layers."""
    
    def __init__(self, layer_name, attention_module, sink_list=None):
        self.layer_name = layer_name
        self.attention_module = attention_module
        self.refiner_attention_maps = []
        self._sink_list = sink_list
    
    def __call__(self, module, input, output):
        """Capture attention weights from the forward pass."""
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # Store attention weights with layer information
                entry = {
                    'layer': self.layer_name,
                    'attention_weights': attn_weights.detach().clone(),
                    'shape': attn_weights.shape
                }
                self.refiner_attention_maps.append(entry)
                if self._sink_list is not None:
                    self._sink_list.append(entry)


class ViTAttentionHook:
    """Hook to capture spatial attention maps from ViT backbone Attention layers."""
    
    def __init__(self, layer_name, attention_module, sink_list=None, extractor=None):
        self.layer_name = layer_name
        self.attention_module = attention_module
        self.attention_maps = []
        self._sink_list = sink_list
        self._original_forward = None
        self._extractor = extractor  # Reference to extractor for immediate saving
        self._save_counter = 0  # Counter for immediate saves
        
    def _hook_forward(self, x):
        """Wrapped forward method that captures attention weights."""
        # Handle both formats: (B, H, W, C) from detectron2 ViT or (B, N, C) from adapter
        if len(x.shape) == 4:
            # Standard detectron2 ViT format: (B, H, W, C)
            B, H, W, C = x.shape
            N = H * W
            is_sequence_format = False
        elif len(x.shape) == 3:
            # Adapter format: (B, N, C)
            B, N, C = x.shape
            # Try to infer H, W from N (assuming square patches)
            # For ViT-L with 16x16 patches: N = H_patches * W_patches
            # We can't know exact H, W without additional info, so we'll use None
            H = W = None
            is_sequence_format = True
        else:
            # Fallback: call original forward
            return self._original_forward(x)
        
        num_heads = self.attention_module.num_heads
        scale = self.attention_module.scale
        
        if is_sequence_format:
            # Adapter format: (B, N, C) -> compute attention in sequence format
            qkv = self.attention_module.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * scale, qkv[1], qkv[2]
            
            # Compute attention scores
            attn = q @ k.transpose(-2, -1)
            
            # Apply softmax to get attention weights
            attn_weights = attn.softmax(dim=-1)
            
            # Store attention weights BEFORE dropout (for visualization purposes)
            # Convert to numpy immediately - this frees GPU tensor memory
            # Shape will be (B, num_heads, N, N)
            # Note: .numpy() creates a view if possible, but detach().cpu() ensures it's separate
            attn_weights_np = attn_weights.detach().cpu().numpy()
            entry = {
                'layer': self.layer_name,
                'attention_weights': attn_weights_np,  # Store as numpy array (no tensor overhead)
                'shape': attn_weights.shape,
                'spatial_shape': None,  # Cannot determine H, W from sequence format
                'sequence_length': N,
                'num_heads': num_heads
            }
            # Save immediately from hook if enabled (prevents accumulation)
            if self._extractor is not None and self._extractor.save_immediately_from_hook:
                # Save directly to disk without accumulating in lists
                self._save_from_hook(entry)
            else:
                # Store entry in sink list (will be cleared after saving)
                if self._sink_list is not None:
                    self._sink_list.append(entry)
                else:
                    self.attention_maps.append(entry)
            # Tensor is freed automatically when we convert to numpy
            # The numpy array is stored in entry, so don't delete it here
            
            # Apply attention dropout (for training/inference consistency)
            if hasattr(self.attention_module, 'attn_drop'):
                attn_weights = self.attention_module.attn_drop(attn_weights)
            
            # Continue with original computation
            x_out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
            x_out = self.attention_module.proj(x_out)
            if hasattr(self.attention_module, 'proj_drop'):
                x_out = self.attention_module.proj_drop(x_out)
            
            return x_out
        else:
            # Standard detectron2 ViT format: (B, H, W, C)
            # Compute QKV
            qkv = self.attention_module.qkv(x).reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * num_heads, N, -1).unbind(0)
            
            # Compute attention scores
            attn = (q * scale) @ k.transpose(-2, -1)
            
            # Add relative positional embeddings if used
            if hasattr(self.attention_module, 'use_rel_pos') and self.attention_module.use_rel_pos:
                try:
                    from detectron2.modeling.backbone.utils import add_decomposed_rel_pos
                    attn = add_decomposed_rel_pos(
                        attn, q, 
                        self.attention_module.rel_pos_h, 
                        self.attention_module.rel_pos_w, 
                        (H, W), (H, W)
                    )
                except ImportError:
                    # If import fails, continue without relative positional embeddings
                    pass
            
            # Apply softmax to get attention weights
            attn_weights = attn.softmax(dim=-1)
            
            # Store attention weights BEFORE dropout (for visualization purposes)
            # Convert to numpy immediately - this frees GPU tensor memory
            attn_weights_np = attn_weights.detach().cpu().numpy()
            entry = {
                'layer': self.layer_name,
                'attention_weights': attn_weights_np,  # Store as numpy array (no tensor overhead)
                'shape': attn_weights.shape,
                'spatial_shape': (H, W),  # Store spatial dimensions
                'num_heads': num_heads
            }
            # Save immediately from hook if enabled (prevents accumulation)
            if self._extractor is not None and self._extractor.save_immediately_from_hook:
                # Save directly to disk without accumulating in lists
                self._save_from_hook(entry)
            else:
                # Store entry in sink list (will be cleared after saving)
                if self._sink_list is not None:
                    self._sink_list.append(entry)
                else:
                    self.attention_maps.append(entry)
            # Tensor is freed automatically when we convert to numpy
            # The numpy array is stored in entry, so don't delete it here
            
            # Apply attention dropout (for training/inference consistency)
            if hasattr(self.attention_module, 'attn_drop'):
                attn_weights = self.attention_module.attn_drop(attn_weights)
            
            # Continue with original computation
            x_out = (attn_weights @ v).view(B, num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x_out = self.attention_module.proj(x_out)
            if hasattr(self.attention_module, 'proj_drop'):
                x_out = self.attention_module.proj_drop(x_out)
            
            return x_out
    
    def _save_from_hook(self, entry):
        """Save attention map directly from hook without accumulating in lists."""
        if self._extractor is None or not self._extractor._output_dir:
            return
        
        try:
            import os
            
            # Get video context if available (set before forward pass)
            video_context = self._extractor._current_video_context
            if video_context is not None:
                video_id = video_context.get('video_id', 'unknown')
                frame_idx = video_context.get('frame_idx', None)
                frame_start = video_context.get('frame_start', None)
                frame_end = video_context.get('frame_end', None)
                window_idx = video_context.get('window_idx', None)
            else:
                # Fallback: no context available
                video_id = 'unknown'
                frame_idx = None
                frame_start = None
                frame_end = None
                window_idx = None
            
            attn_dir = os.path.join(self._extractor._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Build filename with video context if available
            # Extract layer index from layer name (e.g., "backbone.vit_module.blocks.5.attn" -> 5)
            layer_idx = None
            if 'blocks.' in self.layer_name:
                try:
                    layer_idx = int(self.layer_name.split('blocks.')[1].split('.')[0])
                except (ValueError, IndexError):
                    pass
            
            if frame_start is not None and frame_end is not None:
                if layer_idx is not None:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_backbone_vit_layer_{layer_idx}_attn"
                else:
                    # Fallback to old format if layer index can't be extracted
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_layer_{self.layer_name.replace('.', '_')}_attn"
            elif frame_idx is not None:
                if layer_idx is not None:
                    filename = f"video_{video_id}_frame_{frame_idx}_backbone_vit_layer_{layer_idx}_attn"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_layer_{self.layer_name.replace('.', '_')}_attn"
            else:
                if layer_idx is not None:
                    filename = f"video_{video_id}_unknown_backbone_vit_layer_{layer_idx}_attn"
                else:
                    filename = f"video_{video_id}_unknown_layer_{self.layer_name.replace('.', '_')}_attn"
            
            # Add video context to entry
            entry['video_id'] = video_id
            entry['frame_idx'] = frame_idx
            entry['frame_start'] = frame_start
            entry['frame_end'] = frame_end
            entry['window_idx'] = window_idx
            entry['source'] = 'backbone'
            
            # Save synchronously to disk
            if self._extractor is not None:
                self._extractor._save_synchronously(entry, filename, attn_dir)
            else:
                # Fallback: accumulate if extractor not available
                if self._sink_list is not None:
                    self._sink_list.append(entry)
                else:
                    self.attention_maps.append(entry)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save from hook: {e}")
    
    def __call__(self, module, input, output):
        """Forward hook that wraps the attention module's forward method."""
        # This hook is called before the forward pass
        # We'll actually wrap the forward method instead
        pass
    
    def register(self):
        """Register the hook by wrapping the forward method."""
        self._original_forward = self.attention_module.forward
        self.attention_module.forward = self._hook_forward
    
    def unregister(self):
        """Restore the original forward method."""
        if self._original_forward is not None:
            self.attention_module.forward = self._original_forward
            self._original_forward = None


class AttentionExtractor:
    """Class to manage attention extraction hooks and data saving."""
    
    def __init__(self, model, output_dir, max_accumulated_maps=100, 
                 extract_backbone=True, extract_refiner=True, 
                 backbone_layers_to_extract=None, save_immediately_from_hook=False,
                 window_size=None):
        """
        Args:
            model: The model to extract attention from
            output_dir: Directory to save attention maps
            max_accumulated_maps: Maximum attention maps to accumulate before warning
            extract_backbone: If False, skip backbone attention extraction (saves memory)
            extract_refiner: If False, skip refiner attention extraction
            backbone_layers_to_extract: List of layer indices to extract (e.g., [0, 5, 10, 15, 20, 23])
                                        If None, extract all layers. Reduces memory usage.
            save_immediately_from_hook: If True, save directly from hook without accumulating in lists.
                                        This prevents RAM accumulation but requires thread-safe file writing.
        """
        self.model = model
        self.output_dir = output_dir
        self.num_layers = 6  # DVIS-DAQ refiner has 6 layers
        self.refiner_hooks = []  # Hooks for refiner temporal attention
        self.vit_hooks = []  # Separate list for ViT backbone hooks
        self.refiner_attention_data = []  # Refiner attention data
        self.refiner_attention_maps = []  # global sink for refiner temporal attention maps
        self.backbone_attention_maps = []  # global sink for backbone attention maps
        self.video_attention_data = {}  # video_id -> attention_data
        self.current_video_id = None
        self.video_pred_info = None  # Store prediction info for the current video
        self._save_counter = {}  # Track save counter per video to append incrementally
        self._output_dir = output_dir
        self._save_immediately = True  # Save immediately to avoid accumulation
        self._max_accumulated_maps = max_accumulated_maps  # Max entries before forced clear
        self._current_accumulated = 0  # Track current accumulation
        self.extract_backbone = extract_backbone
        self.extract_refiner = extract_refiner
        self.backbone_layers_to_extract = backbone_layers_to_extract  # None = all layers
        self.save_immediately_from_hook = save_immediately_from_hook  # Save directly from hook
        self._current_video_context = None  # Current video_id/frame_idx for hooks to use
        self._window_counter_per_video = {}  # Track window index per video for frame range calculation
        self._window_size = window_size  # Store window_size from config (or None to get from model)
        self._total_frames_per_video = {}  # Track total frames per video (calculated once)
        self._total_windows_per_video = {}  # Track total windows per video (calculated once)
        
        # Register hooks for refiner attention layers and backbone
        self._register_hooks()
        
        # Register a pre-forward hook on the model to set video context before forward pass
        self._model_forward_hook = self.model.register_forward_pre_hook(self._model_forward_pre_hook)
        
        # Register a pre-forward hook on the backbone to update window context each time it's called
        # The backbone is called once per window in segmenter_windows_inference, so this hook will fire
        # for each window, allowing us to track the window index correctly
        if hasattr(self.model, 'backbone'):
            self._backbone_forward_hook = self.model.backbone.register_forward_pre_hook(self._backbone_forward_pre_hook)
    
    def _model_forward_pre_hook(self, module, input):
        """Pre-forward hook on model to extract video_id from inputs and set context."""
        try:
            # Extract video_id from batched_inputs
            if isinstance(input, tuple) and len(input) > 0:
                batched_inputs = input[0]
                if isinstance(batched_inputs, list) and len(batched_inputs) > 0:
                    first_input = batched_inputs[0]
                    if isinstance(first_input, dict):
                        video_id = first_input.get("video_id")
                        frame_idx = first_input.get("frame_idx", None)
                        
                        # Get window_size from config (if provided), otherwise from model, otherwise default to 30
                        window_size = self._window_size
                        if window_size is None:
                            window_size = getattr(module, 'window_size', None)
                        if window_size is None:
                            window_size = 30  # Final fallback default
                        
                        # Set initial video context (window tracking is done in backbone hook)
                        # The model forward hook only extracts video_id, window tracking happens
                        # in the backbone forward hook which fires once per window
                        if video_id is not None:
                            # Calculate and store total frames and windows once per video
                            if "image" in first_input:
                                num_frames = len(first_input["image"])
                                if num_frames is not None and video_id not in self._total_frames_per_video:
                                    self._total_frames_per_video[video_id] = num_frames
                                    # Calculate total windows (ceiling of num_frames/window_size)
                                    self._total_windows_per_video[video_id] = math.ceil(num_frames / window_size)
                            
                            self.set_video_context(
                                video_id=video_id, 
                                frame_idx=frame_idx,
                                frame_start=None,  # Will be set by backbone hook
                                frame_end=None,    # Will be set by backbone hook
                                window_idx=None,   # Will be set by backbone hook
                                forward_pass_id=None
                            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not extract video_id from model input: {e}")
    
    def _backbone_forward_pre_hook(self, module, input):
        """Pre-forward hook on backbone to update window context each time it's called.
        
        The backbone is called once per window in segmenter_windows_inference,
        so this hook fires for each window, allowing us to track the window index correctly.
        """
        try:
            # Get video_id from the current context (set by model forward hook)
            video_id = None
            if self._current_video_context is not None:
                video_id = self._current_video_context.get('video_id')
            
            if video_id is None:
                return
            
            # Get window_size
            window_size = self._window_size
            if window_size is None:
                window_size = getattr(module, 'window_size', None)
            if window_size is None:
                window_size = 30  # Final fallback default
            
            # Get actual number of frames in this window's input tensor
            num_frames_in_this_window = None
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 2:
                    # Input is likely (B, T, C, H, W) or similar
                    num_frames_in_this_window = input_tensor.shape[1] if len(input_tensor.shape) == 5 else input_tensor.shape[0]
            
            # Initialize window counter if needed (new video)
            if video_id not in self._window_counter_per_video:
                self._window_counter_per_video[video_id] = -1
            
            # Increment window index (each backbone forward call is a new window)
            self._window_counter_per_video[video_id] += 1
            
            # Get current window index
            window_idx = self._window_counter_per_video[video_id]
            
            # Calculate frame range for this window
            # frame_start and frame_end are absolute frame indices in the video (inclusive)
            # Window 0: frames 0-30 (31 frames)
            # Window 1: frames 31-61 (31 frames)
            # Window 2: frames 62-92 (31 frames)
            frame_start = window_idx * window_size
            frame_end = (window_idx + 1) * window_size - 1  # Inclusive end, so subtract 1
            
            # Cap frame_end based on total frames in video (for last window which may be shorter)
            total_frames = self._total_frames_per_video.get(video_id)
            if total_frames is not None:
                # frame_end is inclusive, so max is total_frames - 1
                frame_end = min(frame_end, total_frames - 1)
            
            # Debug: print window info
            total_windows = self._total_windows_per_video.get(video_id, None)
            total_info = ""
            if total_frames is not None and total_windows is not None:
                total_info = f", total_frames={total_frames}, total_windows={total_windows}"
            print(f"[WINDOW] video_id={video_id}, window_idx={window_idx}, frames={frame_start}-{frame_end}, num_frames_in_this_window={num_frames_in_this_window}{total_info}", flush=True)
            
            # Update video context for hooks to use (with frame range info)
            if video_id is not None:
                self.set_video_context(
                    video_id=video_id, 
                    frame_idx=self._current_video_context.get('frame_idx') if self._current_video_context else None,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    window_idx=window_idx,
                    forward_pass_id=None
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not update window context in backbone hook: {e}")
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        # Register refiner temporal self-attention hooks
        if self.extract_refiner:
            for name, module in self.model.named_modules():
                if 'transformer_time_self_attention_layers' in name and hasattr(module, 'self_attn'):
                    hook = RefinerAttentionHook(name, module.self_attn, sink_list=self.refiner_attention_maps)
                    module.self_attn.register_forward_hook(hook)
                    self.refiner_hooks.append(hook)
        else:
            print("Skipping refiner attention extraction (disabled)")
        
        # Register ViT backbone spatial attention hooks
        if self.extract_backbone:
            backbone_layer_indices = []
            for name, module in self.model.named_modules():
                # Check if this is a ViT Attention module in the backbone
                # ViT Attention modules have 'qkv' and 'proj' attributes, and are in the backbone
                if (hasattr(module, 'qkv') and hasattr(module, 'proj') and 
                    hasattr(module, 'num_heads') and hasattr(module, 'scale') and
                    ('backbone' in name or 'vit' in name.lower())):
                    # Extract layer index from name (e.g., "backbone.vit_module.blocks.5.attn" -> 5)
                    try:
                        if 'blocks.' in name:
                            layer_idx = int(name.split('blocks.')[1].split('.')[0])
                            # Only register if we want this layer
                            if self.backbone_layers_to_extract is None or layer_idx in self.backbone_layers_to_extract:
                                vit_hook = ViTAttentionHook(name, module, sink_list=self.backbone_attention_maps, extractor=self)
                                vit_hook.register()
                                self.vit_hooks.append(vit_hook)
                                backbone_layer_indices.append(layer_idx)
                                print(f"Registered ViT backbone attention hook for: {name} (layer {layer_idx})")
                            else:
                                print(f"Skipping ViT backbone layer {layer_idx} (not in extraction list)")
                        else:
                            # If we can't extract layer index, register all
                            vit_hook = ViTAttentionHook(name, module, sink_list=self.backbone_attention_maps, extractor=self)
                            vit_hook.register()
                            self.vit_hooks.append(vit_hook)
                            print(f"Registered ViT backbone attention hook for: {name}")
                    except Exception as e:
                        print(f"Warning: Could not register hook for {name}: {e}")
            
            if self.backbone_layers_to_extract is not None:
                print(f"Extracting attention from {len(backbone_layer_indices)} backbone layers: {sorted(backbone_layer_indices)}")
            else:
                print(f"Extracting attention from all {len(backbone_layer_indices)} backbone layers")
        else:
            print("Skipping backbone attention extraction (disabled)")
    
    def set_video_context(self, video_id=None, frame_idx=None, frame_start=None, frame_end=None, window_idx=None, forward_pass_id=None):
        """Set video context for hooks to use when saving directly."""
        self._current_video_context = {
            'video_id': video_id,
            'frame_idx': frame_idx,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'window_idx': window_idx
        }
    
    def clear_video_context(self):
        """Clear video context after forward pass."""
        self._current_video_context = None
    
    def _save_synchronously(self, entry, filename, attn_dir):
        """Save attention map synchronously with progress print."""
        try:
            import os
            import numpy as np
            import json
            from detectron2.utils.file_io import PathManager
            
            arr = entry.get('attention_weights')
            if isinstance(arr, np.ndarray):
                # Define npz_path early so it's available for duplicate checking
                npz_path = os.path.join(attn_dir, f"{filename}.npz")
                
                # Print progress
                print(f"[SAVE] Saving {filename} (shape: {arr.shape})", flush=True)
                
                # Save attention map
                np.savez_compressed(npz_path, attention_weights=arr)
                
                # Save metadata
                meta = {
                    'source': entry.get('source', 'backbone'),
                    'layer': entry.get('layer'),
                    'shape': entry.get('shape'),
                    'spatial_shape': entry.get('spatial_shape'),
                    'num_heads': entry.get('num_heads'),
                    'sequence_length': entry.get('sequence_length'),
                    'video_id': entry.get('video_id'),
                    'frame_idx': entry.get('frame_idx'),
                    'frame_start': entry.get('frame_start'),
                    'frame_end': entry.get('frame_end'),
                    'window_idx': entry.get('window_idx')
                }
                meta_path = os.path.join(attn_dir, f"{filename}.meta.json")
                with PathManager.open(meta_path, "w") as mf:
                    mf.write(json.dumps(meta))
                    mf.flush()
                
                # Free memory
                del arr
                del entry
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save synchronously: {e}")
    
    def clear_attention_maps(self):
        """Clear all captured attention maps and free GPU memory."""
        # Explicitly delete entries to free numpy arrays
        for entry in self.refiner_attention_maps:
            if 'attention_weights' in entry:
                del entry['attention_weights']
        for entry in self.backbone_attention_maps:
            if 'attention_weights' in entry:
                del entry['attention_weights']
        
        self.refiner_attention_maps.clear()
        self.backbone_attention_maps.clear()
        
        for hook in self.refiner_hooks:
            for entry in hook.refiner_attention_maps:
                if 'attention_weights' in entry:
                    del entry['attention_weights']
            hook.refiner_attention_maps.clear()
        
        for hook in self.vit_hooks:
            for entry in hook.attention_maps:
                if 'attention_weights' in entry:
                    del entry['attention_weights']
            hook.attention_maps.clear()
        
        # Force memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_attention_maps_immediately(self, video_id=None, frame_idx=None, force_save=False):
        """Save attention maps immediately to disk to avoid RAM accumulation."""
        if not self._output_dir:
            return
        
        attn_dict = self.get_attention_maps()
        # Check if there's anything to save
        # Note: when save_immediately_from_hook=True, backbone_spatial will be empty
        # because backbone attention is saved directly from hooks
        has_refiner = attn_dict.get('refiner_temporal') and len(attn_dict.get('refiner_temporal', [])) > 0
        has_backbone = attn_dict.get('backbone_spatial') and len(attn_dict.get('backbone_spatial', [])) > 0
        if not has_refiner and not has_backbone:
            return
        
        
        try:
            import os
            import numpy as np
            import json
            from detectron2.utils.file_io import PathManager
            
            attn_dir = os.path.join(self._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Use video_id if provided, otherwise use a temporary identifier
            if video_id is None:
                video_id = "temp"
            
            # Check what types of attention we have
            has_refiner = attn_dict.get('refiner_temporal') and len(attn_dict.get('refiner_temporal', [])) > 0
            has_backbone = attn_dict.get('backbone_spatial') and len(attn_dict.get('backbone_spatial', [])) > 0
            
            # Process and save attention maps
            arrays = {}
            meta = []
            idx = 0
            
            # Refiner attention (temporal)
            refiner_count = 0
            for entry in attn_dict.get('refiner_temporal', []):
                arr = entry.get('attention_weights')
                key = f"attn_{idx}"
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    pass
                arrays[key] = arr
                meta.append({
                    'source': 'refiner',
                    'key': key,
                    'layer': entry.get('layer'),
                    'shape': entry.get('shape'),
                })
                idx += 1
                refiner_count += 1
            if refiner_count > 0:
                print(f"[SAVE] Saving {refiner_count} refiner attention maps for video {video_id}, frame {frame_idx}", flush=True)
            
            # Backbone attention (spatial)
            for entry in attn_dict.get('backbone_spatial', []):
                arr = entry.get('attention_weights')
                key = f"attn_{idx}"
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    pass
                arrays[key] = arr
                meta.append({
                    'source': 'backbone',
                    'key': key,
                    'layer': entry.get('layer'),
                    'shape': entry.get('shape'),
                    'spatial_shape': entry.get('spatial_shape'),
                    'num_heads': entry.get('num_heads'),
                    'sequence_length': entry.get('sequence_length'),
                })
                idx += 1
            
            # Build filename based on what we're saving
            # If saving refiner (temporal) attention only, add "_temporal_refiner_attn" suffix
            # If saving backbone (spatial) attention only, keep original naming
            # If saving both, use a combined name (shouldn't happen in practice)
            if has_refiner and not has_backbone:
                # Only refiner (temporal) attention - add "_temporal_refiner_attn" suffix
                if frame_idx is None:
                    frame_counter = self._save_counter.get(video_id, 0)
                    self._save_counter[video_id] = frame_counter + 1
                    filename = f"video_{video_id}_frame_{frame_counter}_temporal_refiner_attn"
                elif isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    frame_start = min(frame_idx)
                    frame_end = max(frame_idx)
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_temporal_refiner_attn"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_temporal_refiner_attn"
            elif has_backbone and not has_refiner:
                # Only backbone (spatial) attention - keep original naming (will be saved per layer from hooks)
                # This case shouldn't happen here since backbone is saved from hooks
                if frame_idx is None:
                    frame_counter = self._save_counter.get(video_id, 0)
                    self._save_counter[video_id] = frame_counter + 1
                    filename = f"video_{video_id}_frame_{frame_counter}"
                elif isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    frame_start = min(frame_idx)
                    frame_end = max(frame_idx)
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}"
            else:
                # Both types (shouldn't happen in practice, but handle it)
                if frame_idx is None:
                    frame_counter = self._save_counter.get(video_id, 0)
                    self._save_counter[video_id] = frame_counter + 1
                    filename = f"video_{video_id}_frame_{frame_counter}_mixed"
                elif isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    frame_start = min(frame_idx)
                    frame_end = max(frame_idx)
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_mixed"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_mixed"
            
            # Save immediately
            if arrays:
                npz_path = os.path.join(attn_dir, f"{filename}.npz")
                try:
                    np.savez_compressed(npz_path, **arrays)
                except Exception:
                    np.savez(npz_path, **arrays)
                
                meta_path = os.path.join(attn_dir, f"{filename}.meta.json")
                with PathManager.open(meta_path, "w") as mf:
                    mf.write(json.dumps(meta))
                    mf.flush()
            
            # Clear memory immediately
            del arrays, meta
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save attention maps immediately: {e}")
        
        # Always clear after saving to free RAM immediately
        self.clear_attention_maps()
    
    def get_attention_maps(self):
        """Get all captured attention maps (both refiner and backbone)."""
        return {
            'refiner_temporal': self.refiner_attention_maps,
            'backbone_spatial': self.backbone_attention_maps
        }
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        if hasattr(self, '_model_forward_hook') and self._model_forward_hook is not None:
            self._model_forward_hook.remove()
            self._model_forward_hook = None
        for hook in self.vit_hooks:
            hook.unregister()
        self.refiner_hooks.clear()
        self.vit_hooks.clear()