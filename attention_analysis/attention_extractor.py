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


class PredictorCrossAttentionHook:
    """Hook to capture cross-attention weights from predictor (transformer decoder) MultiheadAttention layers."""
    
    def __init__(self, layer_name, layer_idx, attention_module, sink_list=None, extractor=None):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.attention_module = attention_module
        self.attention_maps = []
        self._sink_list = sink_list
        self._extractor = extractor  # Reference to extractor for immediate saving
        self._feature_level = None  # Will be set by predictor forward hook
        self._spatial_shape = None  # Will be set by predictor forward hook (H_feat, W_feat)
    
    def __call__(self, module, input, output):
        """Capture attention weights from the forward pass."""
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # Convert to numpy immediately to free GPU memory
                # Shape: (batch*num_heads, num_queries, H*W) or (num_queries, H*W) depending on batching
                # We need to handle both cases
                attn_weights_np = attn_weights.detach().cpu().numpy()
                
                # Get feature level and spatial shape from extractor (set by predictor forward hook)
                feature_level = None
                spatial_shape = None
                if self._extractor is not None:
                    predictor_info = getattr(self._extractor, '_current_predictor_info', None)
                    if predictor_info is not None:
                        # Calculate feature level: level_index = layer_idx % num_feature_levels
                        num_feature_levels = predictor_info.get('num_feature_levels', 3)
                        feature_level = self.layer_idx % num_feature_levels
                        # Get spatial shape for this feature level
                        size_list = predictor_info.get('size_list', None)
                        if size_list is not None and feature_level < len(size_list):
                            spatial_shape = tuple(size_list[feature_level])  # (H_feat, W_feat)
                
                # Determine actual shape dimensions
                if len(attn_weights_np.shape) == 3:
                    # Shape: (batch*num_heads, num_queries, H*W)
                    batch_heads, num_queries, spatial_length = attn_weights_np.shape
                    # We can't easily separate batch and heads without additional info
                    # Store as-is and let visualization handle it
                elif len(attn_weights_np.shape) == 2:
                    # Shape: (num_queries, H*W) - single batch, single head or already averaged
                    num_queries, spatial_length = attn_weights_np.shape
                    batch_heads = 1
                else:
                    # Unexpected shape, store as-is
                    batch_heads = num_queries = spatial_length = None
                
                entry = {
                    'layer': self.layer_name,
                    'layer_idx': self.layer_idx,
                    'attention_weights': attn_weights_np,  # Store as numpy array
                    'shape': attn_weights.shape,
                    'spatial_shape': spatial_shape,  # (H_feat, W_feat) for this feature level
                    'feature_level': feature_level,  # Which of the 3 feature levels (0, 1, or 2)
                    'spatial_length': spatial_length,  # H*W
                    'num_queries': num_queries,
                    'source': 'predictor_cross_attn'
                }
                
                # Save immediately from hook if enabled
                if self._extractor is not None and self._extractor.save_immediately_from_hook:
                    self._save_from_hook(entry)
                else:
                    # Store entry in sink list
                    if self._sink_list is not None:
                        self._sink_list.append(entry)
                    else:
                        self.attention_maps.append(entry)
    
    def _save_from_hook(self, entry):
        """Save attention map directly from hook without accumulating in lists."""
        if self._extractor is None or not self._extractor._output_dir:
            return
        
        try:
            import os
            import json
            from detectron2.utils.file_io import PathManager
            
            # Get video context if available
            video_context = self._extractor._current_video_context
            if video_context is not None:
                video_id = video_context.get('video_id', 'unknown')
                frame_idx = video_context.get('frame_idx', None)
                frame_start = video_context.get('frame_start', None)
                frame_end = video_context.get('frame_end', None)
                window_idx = video_context.get('window_idx', None)
            else:
                video_id = 'unknown'
                frame_idx = None
                frame_start = None
                frame_end = None
                window_idx = None
            
            # Try to get original image dimensions from image_dimensions.json for mapping
            original_img_height = None
            original_img_width = None
            try:
                inference_dir = os.path.join(self._extractor._output_dir, "inference")
                dims_path = os.path.join(inference_dir, "image_dimensions.json")
                if os.path.exists(dims_path):
                    with PathManager.open(dims_path, "r") as f:
                        dims_dict = json.load(f)
                        video_key = str(video_id)
                        if video_key in dims_dict:
                            dims = dims_dict[video_key]
                            # Use original_image_height/width (after resize, before padding)
                            original_img_height = dims.get('original_image_height')
                            original_img_width = dims.get('original_image_width')
            except Exception:
                pass  # If we can't get dimensions, continue without them
            
            attn_dir = os.path.join(self._extractor._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Build filename with video context
            layer_idx = entry.get('layer_idx', 'unknown')
            feature_level = entry.get('feature_level')
            
            if frame_start is not None and frame_end is not None:
                if feature_level is not None:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
                else:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn_layer_{layer_idx}"
            elif frame_idx is not None:
                if feature_level is not None:
                    filename = f"video_{video_id}_frame_{frame_idx}_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_predictor_cross_attn_layer_{layer_idx}"
            else:
                if feature_level is not None:
                    filename = f"video_{video_id}_unknown_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
                else:
                    filename = f"video_{video_id}_unknown_predictor_cross_attn_layer_{layer_idx}"
            
            # Add video context to entry
            entry['video_id'] = video_id
            entry['frame_idx'] = frame_idx
            entry['frame_start'] = frame_start
            entry['frame_end'] = frame_end
            entry['window_idx'] = window_idx
            
            # Add image dimension info for mapping
            if original_img_height is not None and original_img_width is not None:
                entry['original_image_height'] = original_img_height
                entry['original_image_width'] = original_img_width
                
                # Compute feature stride if we have spatial_shape
                spatial_shape = entry.get('spatial_shape')
                if spatial_shape is not None and len(spatial_shape) == 2:
                    H_feat, W_feat = spatial_shape
                    # Compute stride (downsampling factor)
                    feature_stride_h = original_img_height / H_feat if H_feat > 0 else None
                    feature_stride_w = original_img_width / W_feat if W_feat > 0 else None
                    # They should be approximately equal, use average
                    if feature_stride_h is not None and feature_stride_w is not None:
                        feature_stride = (feature_stride_h + feature_stride_w) / 2.0
                        entry['feature_stride'] = float(feature_stride)
                        entry['feature_stride_h'] = float(feature_stride_h)
                        entry['feature_stride_w'] = float(feature_stride_w)
            
            # Save synchronously to disk
            if self._extractor is not None:
                self._extractor._save_synchronously(entry, filename, attn_dir)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save predictor cross-attention from hook: {e}")


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
        self.predictor_hooks = []  # Hooks for predictor cross-attention
        self.refiner_attention_data = []  # Refiner attention data
        self.refiner_attention_maps = []  # global sink for refiner temporal attention maps
        self.backbone_attention_maps = []  # global sink for backbone attention maps
        self.predictor_attention_maps = []  # global sink for predictor cross-attention maps
        self._current_predictor_info = None  # Store predictor forward info (size_list, num_feature_levels)
        self._query_tracking_maps = {}  # Store query tracking: {video_id: {frame_idx: {seq_id: predictor_query_idx}}}
        self._refiner_id_to_seq_id_maps = {}  # Store refiner_id -> sequence_id mapping: {video_id: {window_key: [seq_id_0, seq_id_1, ...]}}
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
        
        # Register a forward hook on the predictor to capture spatial dimensions
        if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
            self._predictor_forward_hook = self.model.sem_seg_head.predictor.register_forward_hook(self._predictor_forward_hook)
        
        # Register hooks on tracker to capture query mappings
        if hasattr(self.model, 'tracker'):
            self._tracker_match_hook = None
            self._tracker_seq_id_hook = None
            self._tracker_forward_hook = None
            self._setup_tracker_hooks()
        
        # Hook into common_inference to capture final sequence ID mappings
        # This is where the final pred_ids are created from seq_id_list
        # We need to capture the seq_id_list order to map refiner_id (index) -> sequence_id
        if hasattr(self.model, 'common_inference'):
            original_common_inference = self.model.common_inference
            
            def hooked_common_inference(*args, **kwargs):
                # Call original method - this builds seq_id_list and stores it in common_out
                result = original_common_inference(*args, **kwargs)
                
                # Capture seq_id_list from the return value (common_out)
                # This is the order that determines refiner_id -> sequence_id mapping
                if isinstance(result, dict) and 'seq_id_list' in result:
                    seq_id_list = result['seq_id_list']
                    self._store_refiner_id_mapping(seq_id_list)
                
                # After common_inference, capture:
                # 1. The seq_id_list order (refiner_id -> sequence_id mapping) - done above
                # 2. Final sequence ID to track query mappings
                self._capture_final_sequence_mappings()
                
                return result
            
            self.model.common_inference = hooked_common_inference
            self._common_inference_hook = (self.model, original_common_inference)
        else:
            self._common_inference_hook = None
    
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
    
    def _predictor_forward_hook(self, module, input, output):
        """Forward hook on predictor to capture spatial dimensions (size_list) for each feature level.
        
        The predictor forward method receives multi_scale_features (x) as a list of tensors
        with shape (batch, channels, H, W). We extract size_list from x[i].shape[-2:] 
        (same as the predictor does internally at line 374).
        """
        try:
            # The predictor forward receives x (list of multi-scale features) as first argument
            # Format: x is a list of tensors with shape (batch, channels, H, W)
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]  # x is a list of multi-scale feature tensors
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    size_list = []
                    for i, feat in enumerate(x):
                        if hasattr(feat, 'shape') and len(feat.shape) >= 2:
                            # Features should be in format (batch, channels, H, W) before flattening
                            if len(feat.shape) == 4:
                                # (batch, channels, H, W) - extract H, W from last 2 dimensions
                                H, W = feat.shape[-2:]
                                size_list.append((H, W))
                            elif len(feat.shape) == 3:
                                # (H*W, batch, channels) - already flattened, can't get H, W
                                # This shouldn't happen if hook is called at the right time
                                spatial_length, batch, channels = feat.shape
                                size_list.append(None)
                                print(f"[PREDICTOR_HOOK] Warning: Feature {i} is already flattened (shape: {feat.shape}), cannot extract H, W")
                            else:
                                size_list.append(None)
                                print(f"[PREDICTOR_HOOK] Warning: Feature {i} has unexpected shape: {feat.shape}")
                        else:
                            size_list.append(None)
                    
                    num_feature_levels = len(size_list)
                    
                    # Only store predictor info if we got valid dimensions
                    if num_feature_levels > 0 and any(s is not None for s in size_list):
                        self._current_predictor_info = {
                            'size_list': size_list,
                            'num_feature_levels': num_feature_levels
                        }
                        # Debug: print when we successfully capture
                        print(f"[PREDICTOR_HOOK] Captured size_list: {size_list}")
                    else:
                        # Debug: print when we fail to capture
                        print(f"[PREDICTOR_HOOK] Failed to capture valid size_list. Input type: {type(x)}, length: {len(x) if isinstance(x, (list, tuple)) else 'N/A'}")
                        if isinstance(x, (list, tuple)) and len(x) > 0:
                            print(f"[PREDICTOR_HOOK] First feature shape: {x[0].shape if hasattr(x[0], 'shape') else 'N/A'}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not capture predictor info: {e}")
            # Also print for debugging
            print(f"[PREDICTOR_HOOK] Error capturing predictor info: {e}")
    
    def _setup_tracker_hooks(self):
        """Set up hooks on tracker to capture query mappings (track query → predictor query)."""
        try:
            tracker = self.model.tracker
            
            # Hook into match_with_embeds to capture sq_id_for_tq mapping
            original_match_with_embeds = tracker.match_with_embeds
            
            def hooked_match_with_embeds(trc_queries_feat, seg_queries_feat):
                # Call original method
                sq_id_for_tq = original_match_with_embeds(trc_queries_feat, seg_queries_feat)
                
                # Store the mapping (will log only when creating new mappings)
                self._capture_query_mapping(sq_id_for_tq)
                
                return sq_id_for_tq
            
            tracker.match_with_embeds = hooked_match_with_embeds
            self._tracker_match_hook = (tracker, original_match_with_embeds)
            
            # Hook into forward_offline_mode to capture sequence IDs after they're assigned
            # We need to capture last_seq_ids after each frame is processed
            if hasattr(tracker, 'forward_offline_mode'):
                original_forward_offline = tracker.forward_offline_mode
                
                def hooked_forward_offline(*args, **kwargs):
                    # Get parameters
                    resume = kwargs.get('resume', False) if 'resume' in kwargs else (args[4] if len(args) > 4 else False)
                    start_frame_id = kwargs.get('start_frame_id', None) if 'start_frame_id' in kwargs else (args[3] if len(args) > 3 else None)
                    
                    # Get frame_embeds to determine window size
                    # frame_embeds is passed as first positional arg with shape (b, c, t, q)
                    # where t is the number of frames in this window
                    frame_embeds = kwargs.get('frame_embeds', None) if 'frame_embeds' in kwargs else (args[0] if len(args) > 0 else None)
                    
                    # Calculate actual frame range for this window based on start_frame_id and frame_embeds shape
                    if start_frame_id is not None and frame_embeds is not None:
                        if hasattr(frame_embeds, 'shape') and len(frame_embeds.shape) == 4:
                            # frame_embeds shape is (b, c, t, q) where:
                            # b = batch size (typically 1)
                            # c = channels
                            # t = number of frames in this window
                            # q = number of queries
                            T = frame_embeds.shape[2]  # t dimension
                            
                            actual_frame_start = start_frame_id
                            actual_frame_end = start_frame_id + T - 1
                            
                            # Update video context based on actual frame range
                            video_context = self._current_video_context
                            if video_context is not None:
                                video_id = video_context.get('video_id')
                                if video_id is not None:
                                    # Update the video context with the correct frame range
                                    self.set_video_context(
                                        video_id=video_id,
                                        frame_idx=video_context.get('frame_idx'),
                                        frame_start=actual_frame_start,
                                        frame_end=actual_frame_end,
                                        window_idx=video_context.get('window_idx'),
                                        forward_pass_id=None
                                    )
                    
                    # Debug: log when forward_offline_mode is called (once per window)
                    video_context = self._current_video_context
                    if video_context is not None:
                        video_id = video_context.get('video_id')
                        frame_start = video_context.get('frame_start')
                        frame_end = video_context.get('frame_end')
                        print(
                            f"[FORWARD_OFFLINE] video_id={video_id}, frames={frame_start}-{frame_end}, "
                            f"start_frame_id={start_frame_id}, resume={resume}",
                            flush=True
                        )
                    else:
                        print(
                            f"[FORWARD_OFFLINE] video_context=None, start_frame_id={start_frame_id}, resume={resume}",
                            flush=True
                        )
                    
                    # Call original method
                    result = original_forward_offline(*args, **kwargs)
                    
                    # After processing, try to update last_seq_ids for all recent mappings
                    # This ensures we capture sequence IDs that were assigned during processing
                    self._update_all_recent_mappings()
                    
                    return result
                
                tracker.forward_offline_mode = hooked_forward_offline
                self._tracker_forward_hook = (tracker, original_forward_offline)
            else:
                self._tracker_forward_hook = None
            
            # Also hook into inference() method in case it's used instead of forward_offline_mode
            # This might be called for the first window
            if hasattr(tracker, 'inference'):
                original_inference = tracker.inference
                
                def hooked_inference(*args, **kwargs):
                    # Debug: log when inference is called
                    video_context = self._current_video_context
                    resume = kwargs.get('resume', False) if 'resume' in kwargs else (args[4] if len(args) > 4 else False)
                    start_frame_id = kwargs.get('start_frame_id', None) if 'start_frame_id' in kwargs else (args[3] if len(args) > 3 else None)
                    
                    if video_context is not None:
                        video_id = video_context.get('video_id')
                        frame_start = video_context.get('frame_start')
                        frame_end = video_context.get('frame_end')
                        print(
                            f"[TRACKER_INFERENCE] video_id={video_id}, frames={frame_start}-{frame_end}, "
                            f"start_frame_id={start_frame_id}, resume={resume}",
                            flush=True
                        )
                    else:
                        print(
                            f"[TRACKER_INFERENCE] video_context=None, start_frame_id={start_frame_id}, resume={resume}",
                            flush=True
                        )
                    
                    # Call original method
                    result = original_inference(*args, **kwargs)
                    
                    # After processing, try to update last_seq_ids for all recent mappings
                    # This ensures we capture sequence IDs that were assigned during processing
                    self._update_all_recent_mappings()
                    
                    return result
                
                tracker.inference = hooked_inference
                self._tracker_inference_hook = (tracker, original_inference)
            else:
                self._tracker_inference_hook = None
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not set up tracker hooks: {e}")
    
    def _update_all_recent_mappings(self):
        """Update last_seq_ids for all recent mappings after frame processing.
        
        This is called after forward_offline_mode completes, so we can capture
        the final last_seq_ids for the window that was just processed.
        
        We also capture all sequence IDs from video_ins_hub to have a complete
        picture of all sequence IDs that exist.
        
        IMPORTANT: This is called once per window, after all frames in that window
        are processed. We need to update the mapping for the current window.
        """
        try:
            tracker = getattr(self.model, 'tracker', None)
            if tracker is None:
                return
            
            # Update all mappings for the current video
            video_context = self._current_video_context
            if video_context is None:
                return
            
            video_id = video_context.get('video_id')
            if video_id is None:
                return
            
            if video_id not in self._query_tracking_maps:
                return
            
            frames = self._query_tracking_maps[video_id]
            if not frames:
                return
            
            # Get frame_start and frame_end from context to find the right window
            frame_start = video_context.get('frame_start')
            frame_end = video_context.get('frame_end')
            
            if frame_start is not None and frame_end is not None:
                frame_key = f"frames_{frame_start}_{frame_end}"
            else:
                # Fallback: use the most recently added frame_key
                frame_keys = list(frames.keys())
                if not frame_keys:
                    return
                frame_key = frame_keys[-1]
            
            # If the mapping doesn't exist yet, create it
            # This can happen for the first window if match_with_embeds wasn't called for frame 0
            # But match_with_embeds should be called for frames 1-99, so we should have sq_id_for_tq
            # If we don't have it, we'll create a placeholder and try to get sq_id_for_tq from the tracker
            if frame_key not in frames:
                # Try to get sq_id_for_tq from the most recent match_with_embeds call
                # We can't directly access it, but we can create a placeholder
                # The sq_id_for_tq will be captured when match_with_embeds is called for frames 1-99
                frames[frame_key] = {
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'frame_idx': video_context.get('frame_idx'),
                    'sq_id_for_tq': None,  # Will be set when match_with_embeds is called
                    'num_track_queries': 0
                }
            else:
                # If mapping exists but sq_id_for_tq is None, we might need to wait for match_with_embeds
                # But since this is called after forward_offline_mode, match_with_embeds should have been called
                # So if sq_id_for_tq is still None, it means match_with_embeds wasn't called for this window
                # This can happen for the first frame of the first window
                pass
            
            # Try to capture last_seq_ids (active track queries for last frame)
            if hasattr(tracker, 'last_seq_ids'):
                last_seq_ids = tracker.last_seq_ids
                if last_seq_ids is not None and len(last_seq_ids) > 0:
                    frames[frame_key]['last_seq_ids'] = (
                        last_seq_ids if isinstance(last_seq_ids, list) else last_seq_ids.tolist()
                    )
            
            # Also capture all sequence IDs from video_ins_hub
            # This gives us all sequence IDs that exist, not just active ones
            # The refiner_id in results comes from video_ins_hub, so this helps us match
            if hasattr(tracker, 'video_ins_hub'):
                all_seq_ids = list(tracker.video_ins_hub.keys())
                if all_seq_ids:
                    frames[frame_key]['all_seq_ids_from_hub'] = (
                        sorted(all_seq_ids) if isinstance(all_seq_ids[0], (int, float)) else sorted([int(x) for x in all_seq_ids])
                    )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not update recent mappings: {e}")
    
    def _store_refiner_id_mapping(self, seq_id_list):
        """Store the refiner_id -> sequence_id mapping for the current window."""
        try:
            video_context = self._current_video_context
            if video_context is None:
                return
            
            video_id = video_context.get('video_id')
            frame_start = video_context.get('frame_start')
            frame_end = video_context.get('frame_end')
            
            if video_id is None:
                return
            
            if video_id not in self._refiner_id_to_seq_id_maps:
                self._refiner_id_to_seq_id_maps[video_id] = {}
            
            # Use frame range as key
            if frame_start is not None and frame_end is not None:
                window_key = f"frames_{frame_start}_{frame_end}"
            else:
                window_key = "unknown"
            
            # Store the mapping: refiner_id (index) -> sequence_id
            # seq_id_list[i] is the sequence_id for refiner_id = i
            self._refiner_id_to_seq_id_maps[video_id][window_key] = (
                seq_id_list if isinstance(seq_id_list, list) else seq_id_list.tolist()
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not store refiner_id mapping: {e}")
    
    def _capture_final_sequence_mappings(self):
        """Capture final sequence ID mappings from video_ins_hub at the end of common_inference.
        
        This is called after all windows are processed, so we can capture the final
        mapping between sequence IDs and their positions in the final output.
        """
        try:
            tracker = getattr(self.model, 'tracker', None)
            if tracker is None:
                return
            
            if not hasattr(tracker, 'video_ins_hub'):
                return
            
            video_context = self._current_video_context
            if video_context is None:
                return
            
            video_id = video_context.get('video_id')
            if video_id is None:
                return
            
            if video_id not in self._query_tracking_maps:
                return
            
            # Get all sequence IDs from video_ins_hub (these are the refiner_ids in final output)
            all_seq_ids = list(tracker.video_ins_hub.keys())
            
            # For each window, try to map sequence IDs to track query positions
            # We need to find which track query position each sequence ID had in each window
            frames = self._query_tracking_maps[video_id]
            
            for frame_key, mapping in frames.items():
                last_seq_ids = mapping.get('last_seq_ids')
                sq_id_for_tq = mapping.get('sq_id_for_tq')
                
                if last_seq_ids is None or sq_id_for_tq is None:
                    continue
                
                # Create a mapping: seq_id -> (track_query_idx, predictor_query_id)
                # IMPORTANT: We need sq_id_for_tq to create this mapping
                # If sq_id_for_tq is None, we can't create the mapping, but we can still store last_seq_ids
                seq_id_to_query = {}
                if sq_id_for_tq is not None:
                    for track_query_idx, seq_id in enumerate(last_seq_ids):
                        if track_query_idx < len(sq_id_for_tq):
                            predictor_query_id = int(sq_id_for_tq[track_query_idx])
                            seq_id_to_query[seq_id] = {
                                'track_query_idx': track_query_idx,
                                'predictor_query_id': predictor_query_id
                            }
                    
                    # Store this mapping for later lookup
                    mapping['seq_id_to_query_mapping'] = seq_id_to_query
                else:
                    # If sq_id_for_tq is None, we can't create the mapping yet
                    # This can happen for the first window if match_with_embeds wasn't called
                    # We'll store last_seq_ids anyway, and the mapping will be created when sq_id_for_tq is available
                    import logging
                    logging.getLogger(__name__).debug(
                        f"Window {frame_key}: sq_id_for_tq is None, cannot create seq_id_to_query_mapping. "
                        f"last_seq_ids={last_seq_ids is not None}"
                    )
                
                mapping['all_seq_ids_from_hub'] = sorted(all_seq_ids) if all_seq_ids else []
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not capture final sequence mappings: {e}")
    
    def _capture_query_mapping(self, sq_id_for_tq):
        """Capture the mapping from track query indices to predictor query indices.
        
        Args:
            sq_id_for_tq: Tensor of shape (num_track_queries + 1,) where sq_id_for_tq[i] 
                         is the predictor query index for track query i (last is background)
        """
        try:
            # Get current video context
            video_context = self._current_video_context
            if video_context is None:
                print("[QUERY_MAPPING] video_context is None, skipping", flush=True)
                return
            
            video_id = video_context.get('video_id')
            frame_idx = video_context.get('frame_idx')
            frame_start = video_context.get('frame_start')
            frame_end = video_context.get('frame_end')
            
            if video_id is None:
                print("[QUERY_MAPPING] video_id is None, skipping", flush=True)
                return
            
            # Convert to numpy for storage
            if isinstance(sq_id_for_tq, torch.Tensor):
                sq_id_for_tq_np = sq_id_for_tq.detach().cpu().numpy()
            else:
                sq_id_for_tq_np = sq_id_for_tq
            
            # Store mapping
            # Use frame_start-frame_end as key if available, otherwise frame_idx
            # Note: match_with_embeds is called per frame, but we want to store per window
            # We'll use the window frame range as the key, and update it with the latest frame's data
            if frame_start is not None and frame_end is not None:
                frame_key = f"frames_{frame_start}_{frame_end}"
            elif frame_idx is not None:
                # Try to infer window from frame_idx
                window_size = self._window_size or 30
                window_start = (frame_idx // window_size) * window_size
                window_end = min(window_start + window_size - 1, self._total_frames_per_video.get(video_id, window_start + window_size - 1) - 1)
                frame_key = f"frames_{window_start}_{window_end}"
            else:
                frame_key = "unknown"
            
            if video_id not in self._query_tracking_maps:
                self._query_tracking_maps[video_id] = {}
            
            # Check if we already have a mapping for this window
            # If so, we'll update it (since match_with_embeds is called per frame, we want the last one)
            is_new_mapping = frame_key not in self._query_tracking_maps[video_id]
            
            if is_new_mapping:
                # Create new entry
                mapping_entry = {
                    'sq_id_for_tq': sq_id_for_tq_np.tolist(),  # track_query_idx -> predictor_query_idx
                    'num_track_queries': len(sq_id_for_tq_np),
                    'frame_idx': frame_idx,
                    'frame_start': frame_start,
                    'frame_end': frame_end
                }
                print(
                    f"[QUERY_MAPPING] Created NEW mapping for video_id={video_id}, "
                    f"frame_key={frame_key}, frame_start={frame_start}, frame_end={frame_end}, "
                    f"sq_id_for_tq.shape={sq_id_for_tq.shape if hasattr(sq_id_for_tq, 'shape') else 'N/A'}",
                    flush=True
                )
            else:
                # Update existing entry (keep any existing last_seq_ids if we don't have new ones yet)
                existing = self._query_tracking_maps[video_id][frame_key]
                mapping_entry = existing.copy()
                mapping_entry['sq_id_for_tq'] = sq_id_for_tq_np.tolist()
                mapping_entry['num_track_queries'] = len(sq_id_for_tq_np)
                # Update frame info if we have more specific info
                if frame_start is not None:
                    mapping_entry['frame_start'] = frame_start
                if frame_end is not None:
                    mapping_entry['frame_end'] = frame_end
                if frame_idx is not None:
                    mapping_entry['frame_idx'] = frame_idx
                # Don't log every update - too verbose (called once per frame)
            
            # Also try to capture sequence IDs and activated queries if available
            # Note: We need to capture this AFTER the sequence IDs are assigned, which happens
            # after match_with_embeds is called. So we'll capture it in a delayed way.
            # For now, we'll try to get it, but it might not be available yet.
            tracker = getattr(self.model, 'tracker', None)
            if tracker is not None:
                # Capture last_seq_ids (sequence IDs for activated track queries)
                # This is set after processing each frame, so we need to capture it at the right time
                if hasattr(tracker, 'last_seq_ids'):
                    last_seq_ids = tracker.last_seq_ids
                    if last_seq_ids is not None and len(last_seq_ids) > 0:
                        # last_seq_ids is a list where last_seq_ids[k] = seq_id for track query k
                        # But this is only for activated queries, so indices may not align with sq_id_for_tq
                        mapping_entry['last_seq_ids'] = (
                            last_seq_ids if isinstance(last_seq_ids, list) else last_seq_ids.tolist()
                        )
                
                # Capture track_queries to understand which are activated
                # The track_queries tensor shape tells us how many are active
                if hasattr(tracker, 'track_queries') and tracker.track_queries is not None:
                    num_active_track_queries = tracker.track_queries.shape[0]
                    mapping_entry['num_active_track_queries'] = int(num_active_track_queries)
            
            # Store the mapping (we'll update last_seq_ids later if needed)
            self._query_tracking_maps[video_id][frame_key] = mapping_entry
            
            # Also try to update last_seq_ids after a short delay to catch sequence ID assignments
            # This is a workaround - ideally we'd hook into the sequence ID assignment directly
            # For now, we'll try to update it when we can
            self._try_update_last_seq_ids(video_id, frame_key)
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not capture query mapping: {e}")
    
    def _try_update_last_seq_ids(self, video_id, frame_key):
        """Try to update last_seq_ids for a mapping entry after sequence IDs are assigned."""
        try:
            if video_id not in self._query_tracking_maps:
                return
            if frame_key not in self._query_tracking_maps[video_id]:
                return
            
            tracker = getattr(self.model, 'tracker', None)
            if tracker is None:
                return
            
            # Try to get last_seq_ids if available
            if hasattr(tracker, 'last_seq_ids'):
                last_seq_ids = tracker.last_seq_ids
                if last_seq_ids is not None and len(last_seq_ids) > 0:
                    # Update the mapping entry
                    self._query_tracking_maps[video_id][frame_key]['last_seq_ids'] = (
                        last_seq_ids if isinstance(last_seq_ids, list) else last_seq_ids.tolist()
                    )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not update last_seq_ids: {e}")
    
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
        
        # Register predictor cross-attention hooks
        if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
            predictor = self.model.sem_seg_head.predictor
            if hasattr(predictor, 'transformer_cross_attention_layers'):
                predictor_layer_indices = []
                for i, cross_attn_layer in enumerate(predictor.transformer_cross_attention_layers):
                    if hasattr(cross_attn_layer, 'multihead_attn'):
                        layer_name = f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}"
                        hook = PredictorCrossAttentionHook(
                            layer_name, i, cross_attn_layer.multihead_attn,
                            sink_list=self.predictor_attention_maps,
                            extractor=self
                        )
                        cross_attn_layer.multihead_attn.register_forward_hook(hook)
                        self.predictor_hooks.append(hook)
                        predictor_layer_indices.append(i)
                        print(f"Registered predictor cross-attention hook for layer {i}")
                
                print(f"Extracting attention from {len(predictor_layer_indices)} predictor cross-attention layers: {predictor_layer_indices}")
            else:
                print("Warning: Could not find transformer_cross_attention_layers in predictor")
        else:
            print("Skipping predictor cross-attention extraction (predictor not found)")
    
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
                    'layer_idx': entry.get('layer_idx'),
                    'feature_level': entry.get('feature_level'),
                    'spatial_length': entry.get('spatial_length'),
                    'num_queries': entry.get('num_queries'),
                    'original_image_height': entry.get('original_image_height'),
                    'original_image_width': entry.get('original_image_width'),
                    'feature_stride': entry.get('feature_stride'),
                    'feature_stride_h': entry.get('feature_stride_h'),
                    'feature_stride_w': entry.get('feature_stride_w'),
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
        for entry in self.predictor_attention_maps:
            if 'attention_weights' in entry:
                del entry['attention_weights']
        
        self.refiner_attention_maps.clear()
        self.backbone_attention_maps.clear()
        self.predictor_attention_maps.clear()
        
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
        
        for hook in self.predictor_hooks:
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
        # Note: when save_immediately_from_hook=True, backbone_spatial and predictor_cross_attn will be empty
        # because they are saved directly from hooks
        has_refiner = attn_dict.get('refiner_temporal') and len(attn_dict.get('refiner_temporal', [])) > 0
        has_backbone = attn_dict.get('backbone_spatial') and len(attn_dict.get('backbone_spatial', [])) > 0
        has_predictor = attn_dict.get('predictor_cross_attn') and len(attn_dict.get('predictor_cross_attn', [])) > 0
        if not has_refiner and not has_backbone and not has_predictor:
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
            
            # Predictor cross-attention
            predictor_count = 0
            for entry in attn_dict.get('predictor_cross_attn', []):
                arr = entry.get('attention_weights')
                key = f"attn_{idx}"
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    pass
                arrays[key] = arr
                meta.append({
                    'source': 'predictor_cross_attn',
                    'key': key,
                    'layer': entry.get('layer'),
                    'layer_idx': entry.get('layer_idx'),
                    'shape': entry.get('shape'),
                    'spatial_shape': entry.get('spatial_shape'),
                    'feature_level': entry.get('feature_level'),
                    'spatial_length': entry.get('spatial_length'),
                    'num_queries': entry.get('num_queries'),
                })
                idx += 1
                predictor_count += 1
            if predictor_count > 0:
                print(f"[SAVE] Saving {predictor_count} predictor cross-attention maps for video {video_id}, frame {frame_idx}", flush=True)
            
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
        """Get all captured attention maps (refiner, backbone, and predictor)."""
        return {
            'refiner_temporal': self.refiner_attention_maps,
            'backbone_spatial': self.backbone_attention_maps,
            'predictor_cross_attn': self.predictor_attention_maps
        }
    
    def save_query_tracking_maps(self):
        """Save query tracking maps to disk."""
        if not self._output_dir or not self._query_tracking_maps:
            return
        
        try:
            import os
            import json
            from detectron2.utils.file_io import PathManager
            
            attn_dir = os.path.join(self._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            tracking_path = os.path.join(attn_dir, "query_tracking_maps.json")
            
            # Convert numpy arrays to lists for JSON serialization
            tracking_dict = {}
            for video_id, frames in self._query_tracking_maps.items():
                tracking_dict[str(video_id)] = {}
                for frame_key, mapping in frames.items():
                    tracking_dict[str(video_id)][frame_key] = mapping
            
            with PathManager.open(tracking_path, "w") as f:
                f.write(json.dumps(tracking_dict, indent=2))
                f.flush()
            
            print(f"[SAVE] Saved query tracking maps for {len(self._query_tracking_maps)} videos to {tracking_path}", flush=True)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save query tracking maps: {e}")
    
    def get_predictor_query_for_refiner_id(self, video_id, refiner_id, frame_idx=None):
        """Get predictor query index for a given refiner_id (index in refiner output).
        
        Args:
            video_id: Video ID
            refiner_id: Refiner ID (index in refiner output, NOT sequence ID)
            frame_idx: Frame index (optional, uses last frame if not provided)
        
        Returns:
            predictor_query_idx: Predictor query index, or None if not found
        """
        # First, convert refiner_id (index) to sequence_id
        sequence_id = self._get_sequence_id_from_refiner_id(video_id, refiner_id)
        if sequence_id is None:
            return None
        
        # Now use sequence_id to find predictor query (same as before)
        if video_id not in self._query_tracking_maps:
            return None
        
        frames = self._query_tracking_maps[video_id]
        
        # Find the frame mapping
        if frame_idx is not None:
            # Try exact frame first
            frame_key = f"frame_{frame_idx}"
            if frame_key not in frames:
                # Try frame range
                frame_key = f"frames_{frame_idx}_{frame_idx}"
        else:
            # Use the last frame available
            frame_keys = sorted([k for k in frames.keys() if k.startswith("frame")])
            if not frame_keys:
                return None
            frame_key = frame_keys[-1]
        
        if frame_key not in frames:
            return None
        
        mapping = frames[frame_key]
        last_seq_ids = mapping.get('last_seq_ids')
        sq_id_for_tq = mapping.get('sq_id_for_tq')
        seq_id_to_query = mapping.get('seq_id_to_query_mapping')
        
        # Try seq_id_to_query_mapping first (most reliable)
        if seq_id_to_query is not None and sequence_id in seq_id_to_query:
            query_info = seq_id_to_query[sequence_id]
            return query_info['predictor_query_id']
        
        # Fallback: try last_seq_ids
        if last_seq_ids is None or sq_id_for_tq is None:
            return None
        
        # Find track query index for this sequence ID
        # Note: last_seq_ids[k] = seq_id for track query k (only for activated queries)
        # sq_id_for_tq[k] = predictor query index for track query k
        try:
            # Find which track query index has this sequence ID
            track_query_idx = last_seq_ids.index(sequence_id)
            # Get predictor query index from the mapping
            if track_query_idx < len(sq_id_for_tq):
                predictor_query_idx = sq_id_for_tq[track_query_idx]
                return int(predictor_query_idx)
        except (ValueError, IndexError):
            pass
        
        return None
    
    def get_all_predictor_queries_for_refiner_id(self, video_id, refiner_id):
        """Get all predictor query indices for a given refiner_id across all windows.
        
        Args:
            video_id: Video ID
            refiner_id: Refiner ID (index in refiner output, NOT sequence ID)
        
        Returns:
            List of dicts with keys: 'predictor_query_id', 'frame_start', 'frame_end', 'frame_key'
            Returns empty list if not found
        """
        # First, convert refiner_id (index) to sequence_id
        sequence_id = self._get_sequence_id_from_refiner_id(video_id, refiner_id)
        if sequence_id is None:
            return []
        
        # Now use sequence_id to find predictor queries (same as before)
        if video_id not in self._query_tracking_maps:
            return []
        
        frames = self._query_tracking_maps[video_id]
        results = []
        
        # Check all windows/frames
        for frame_key, mapping in frames.items():
            last_seq_ids = mapping.get('last_seq_ids')
            sq_id_for_tq = mapping.get('sq_id_for_tq')
            seq_id_to_query = mapping.get('seq_id_to_query_mapping')
            
            # Skip if we don't have sq_id_for_tq (this means match_with_embeds wasn't called for this window)
            # This can happen for the first window if match_with_embeds wasn't called for any frame
            # But it should be called for frames 1-99, so this shouldn't happen unless there's a bug
            if sq_id_for_tq is None:
                # Debug: log why we're skipping this window
                print(
                    f"[QUERY_LOOKUP] Skipping window {frame_key} for video {video_id}, refiner_id {refiner_id}: "
                    f"sq_id_for_tq is None. last_seq_ids={last_seq_ids is not None}, "
                    f"seq_id_to_query={seq_id_to_query is not None}",
                    flush=True
                )
                continue
            
            # Try to find sequence_id using the seq_id_to_query_mapping first (most reliable)
            found = False
            predictor_query_id = None
            
            if seq_id_to_query is not None and sequence_id in seq_id_to_query:
                query_info = seq_id_to_query[sequence_id]
                predictor_query_id = query_info['predictor_query_id']
                found = True
            elif last_seq_ids is not None:
                # Fallback: try last_seq_ids (active track queries for last frame)
                try:
                    track_query_idx = last_seq_ids.index(sequence_id)
                    if track_query_idx < len(sq_id_for_tq):
                        predictor_query_id = int(sq_id_for_tq[track_query_idx])
                        found = True
                except (ValueError, IndexError):
                    pass
            
            # If not found, skip this window
            if not found:
                continue
            
            # Get frame information
            frame_start = mapping.get('frame_start')
            frame_end = mapping.get('frame_end')
            frame_idx = mapping.get('frame_idx')
            
            results.append({
                'predictor_query_id': predictor_query_id,
                'frame_start': frame_start,
                'frame_end': frame_end,
                'frame_idx': frame_idx,
                'frame_key': frame_key
            })
        
        # Sort by frame_start (or frame_idx if frame_start is None)
        results.sort(key=lambda x: x.get('frame_start') if x.get('frame_start') is not None else (x.get('frame_idx') if x.get('frame_idx') is not None else 0))
        
        return results
    
    def _get_sequence_id_from_refiner_id(self, video_id, refiner_id):
        """Convert refiner_id (index) to sequence_id.
        
        Args:
            video_id: Video ID
            refiner_id: Index in refiner output (0, 1, 2, ...)
        
        Returns:
            sequence_id: The sequence ID corresponding to this refiner_id, or None
        """
        if video_id not in self._refiner_id_to_seq_id_maps:
            return None
        
        video_maps = self._refiner_id_to_seq_id_maps[video_id]
        
        # Try to find the sequence_id in any window
        # The seq_id_list order should be consistent across windows (from video_ins_hub iteration)
        # But we'll check all windows and use the first one that has enough entries
        for window_key, seq_id_list in video_maps.items():
            if isinstance(seq_id_list, list) and refiner_id < len(seq_id_list):
                return int(seq_id_list[refiner_id])
        
        # If not found in any window, return None
        return None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        if hasattr(self, '_model_forward_hook') and self._model_forward_hook is not None:
            self._model_forward_hook.remove()
            self._model_forward_hook = None
        if hasattr(self, '_backbone_forward_hook') and self._backbone_forward_hook is not None:
            self._backbone_forward_hook.remove()
            self._backbone_forward_hook = None
        if hasattr(self, '_predictor_forward_hook') and self._predictor_forward_hook is not None:
            self._predictor_forward_hook.remove()
            self._predictor_forward_hook = None
        # Restore tracker methods if hooked
        if hasattr(self, '_tracker_match_hook') and self._tracker_match_hook is not None:
            tracker, original_method = self._tracker_match_hook
            tracker.match_with_embeds = original_method
            self._tracker_match_hook = None
        if hasattr(self, '_tracker_forward_hook') and self._tracker_forward_hook is not None:
            tracker, original_method = self._tracker_forward_hook
            tracker.forward_offline_mode = original_method
            self._tracker_forward_hook = None
        if hasattr(self, '_tracker_inference_hook') and self._tracker_inference_hook is not None:
            tracker, original_method = self._tracker_inference_hook
            tracker.inference = original_method
            self._tracker_inference_hook = None
        if hasattr(self, '_common_inference_hook') and self._common_inference_hook is not None:
            model, original_method = self._common_inference_hook
            model.common_inference = original_method
            self._common_inference_hook = None
        if hasattr(self, '_run_window_hook') and self._run_window_hook is not None:
            model, original_method = self._run_window_hook
            model.run_window_inference = original_method
            self._run_window_hook = None
        for hook in self.vit_hooks:
            hook.unregister()
        self.refiner_hooks.clear()
        self.vit_hooks.clear()
        self.predictor_hooks.clear()