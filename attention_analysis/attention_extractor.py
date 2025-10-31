import torch

class AttentionHook:
    """Hook to capture attention weights from MultiheadAttention layers."""
    
    def __init__(self, layer_name, attention_module, sink_list=None):
        self.layer_name = layer_name
        self.attention_module = attention_module
        self.attention_maps = []
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
                self.attention_maps.append(entry)
                if self._sink_list is not None:
                    self._sink_list.append(entry)

class AttentionExtractor:
    """Class to manage attention extraction hooks and data saving."""
    
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.num_layers = 6  # DVIS-DAQ refiner has 6 layers
        self.hooks = []
        self.attention_data = []
        self.attention_maps = []  # global sink for all registered hooks
        self.video_attention_data = {}  # video_id -> attention_data
        self.current_video_id = None
        self.video_pred_info = None  # Store prediction info for the current video
        
        # Register hooks for refiner attention layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        # Register refiner temporal self-attention hooks
        for name, module in self.model.named_modules():
            if 'transformer_time_self_attention_layers' in name and hasattr(module, 'self_attn'):
                hook = AttentionHook(name, module.self_attn, sink_list=self.attention_maps)
                module.self_attn.register_forward_hook(hook)
                self.hooks.append(hook)