# Pixel Decoder Cross-Attention Spatial Location Mapping

## Overview

The pixel decoder cross-attention maps show how **object queries** attend to **spatial features** from the pixel decoder. Unlike backbone ViT attention (which uses 16×16 patches directly from the image), pixel decoder features are multi-scale and have been processed through FPN and deformable transformer, requiring a different mapping approach.

## Architecture Flow

### 1. Pixel Decoder Output
The pixel decoder (`model.sem_seg_head.pixel_decoder`) processes backbone features and outputs:
- `multi_scale_features`: A list of **3 feature maps** at different scales
- Each feature map has shape `(batch, channels, H_feat, W_feat)`
- These are the spatial features that object queries will attend to

### 2. Predictor Cross-Attention
The predictor (`model.sem_seg_head.predictor`) is a `MultiScaleMaskedTransformerDecoder` with:
- `num_feature_levels = 3` (always uses 3 scales)
- `transformer_cross_attention_layers`: List of cross-attention layers
- Each layer attends to a different feature level: `level_index = layer_idx % 3`

### 3. Attention Weight Shape
When you extract cross-attention from layer `i`:
- **Shape**: `(batch_size, num_heads, num_queries, H_feat * W_feat)`
- Where `H_feat, W_feat` are the spatial dimensions of the feature level used by that layer
- The feature level cycles: layer 0→level 0, layer 1→level 1, layer 2→level 2, layer 3→level 0, etc.

## Spatial Dimensions at Each Feature Level

### Feature Level Strides
The 3 feature levels typically have different spatial resolutions:
- **Level 0**: Highest resolution (e.g., stride 8x from original image)
- **Level 1**: Medium resolution (e.g., stride 16x from original image)  
- **Level 2**: Lowest resolution (e.g., stride 32x from original image)

**Note**: The exact strides depend on your model configuration (`MODEL.SEM_SEG_HEAD.COMMON_STRIDE` and backbone output strides).

### Capturing Spatial Dimensions
When extracting attention, you need to capture:
1. **Feature level index** (`level_index = layer_idx % 3`)
2. **Spatial dimensions** `(H_feat, W_feat)` for that level
3. **Original image dimensions** for mapping back

## Mapping Feature Coordinates to Image Coordinates

### Step 1: Reshape Attention to 2D Grid
```python
# Attention weights shape: (batch, num_heads, num_queries, H_feat * W_feat)
# For a specific query and head:
attn_1d = attention_weights[batch_idx, head_idx, query_idx, :]  # Shape: (H_feat * W_feat,)
attn_2d = attn_1d.reshape(H_feat, W_feat)  # Shape: (H_feat, W_feat)
```

### Step 2: Map Feature Coordinates to Image Coordinates
Each feature location `(h_feat, w_feat)` corresponds to a region in the original image:

```python
def feature_coord_to_image_coord(h_feat, w_feat, H_feat, W_feat, 
                                   original_img_height, original_img_width,
                                   feature_stride):
    """
    Map feature-level coordinates to original image coordinates.
    
    Args:
        h_feat, w_feat: Feature-level coordinates (0 to H_feat-1, 0 to W_feat-1)
        H_feat, W_feat: Feature map dimensions
        original_img_height, original_img_width: Original image dimensions
        feature_stride: Downsampling factor (e.g., 8, 16, or 32)
    
    Returns:
        (x_min, y_min, x_max, y_max): Bounding box in original image coordinates
    """
    # Feature maps are downsampled by feature_stride
    # Each feature location corresponds to a feature_stride × feature_stride region
    
    # Map to image coordinates (accounting for downsampling)
    x_min = w_feat * feature_stride
    y_min = h_feat * feature_stride
    x_max = x_min + feature_stride
    y_max = y_min + feature_stride
    
    # Clamp to image boundaries
    x_min = max(0, min(x_min, original_img_width))
    y_min = max(0, min(y_min, original_img_height))
    x_max = max(0, min(x_max, original_img_width))
    y_max = max(0, min(y_max, original_img_height))
    
    return (x_min, y_min, x_max, y_max)
```

### Step 3: Create Attention Heatmap Overlay
```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def overlay_attention_on_image(image, attention_2d, feature_stride, 
                               original_img_height, original_img_width):
    """
    Overlay attention heatmap on original image.
    
    Args:
        image: PIL Image or numpy array of original image
        attention_2d: 2D attention map (H_feat, W_feat)
        feature_stride: Downsampling factor for this feature level
        original_img_height, original_img_width: Original image dimensions
    """
    H_feat, W_feat = attention_2d.shape
    
    # Upsample attention to image resolution
    # Method 1: Simple upsampling (nearest neighbor)
    from scipy.ndimage import zoom
    scale_h = original_img_height / H_feat
    scale_w = original_img_width / W_feat
    attention_upsampled = zoom(attention_2d, (scale_h, scale_w), order=0)
    
    # Method 2: More accurate - map each feature location to its image region
    attention_map = np.zeros((original_img_height, original_img_width))
    for h_feat in range(H_feat):
        for w_feat in range(W_feat):
            x_min, y_min, x_max, y_max = feature_coord_to_image_coord(
                h_feat, w_feat, H_feat, W_feat,
                original_img_height, original_img_width, feature_stride
            )
            attention_map[y_min:y_max, x_min:x_max] = attention_2d[h_feat, w_feat]
    
    # Normalize attention for visualization
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Create overlay
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    im = ax.imshow(attention_map, alpha=0.5, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Object Query Attention (Feature Level, Stride {feature_stride}x)')
    plt.colorbar(im, ax=ax)
    return fig
```

## Implementation Details for Extraction

### What to Capture in the Hook

When extracting pixel decoder cross-attention, you need to save:

1. **Attention weights**: `(batch, num_heads, num_queries, H_feat * W_feat)`
2. **Feature level index**: `level_index = layer_idx % num_feature_levels`
3. **Spatial dimensions**: `(H_feat, W_feat)` for this feature level
4. **Feature stride**: Downsampling factor for this level (need to compute from config or capture)

### Capturing Spatial Dimensions

The predictor forward method captures `size_list`:
```python
for i in range(self.num_feature_levels):
    size_list.append(x[i].shape[-2:])  # (H_feat, W_feat) for each level
```

You can capture this by:
1. **Option A**: Hook into predictor forward and capture `size_list`
2. **Option B**: Extract from the input `x` (multi_scale_features) dimensions
3. **Option C**: Compute from original image dimensions and feature stride

### Feature Stride Calculation

The feature stride for each level depends on:
- Backbone output strides (e.g., res2=4x, res3=8x, res4=16x, res5=32x)
- Pixel decoder processing (FPN, deformable transformer)
- `COMMON_STRIDE` config (typically 4 or 8)

**Typical mapping** (for Mask2Former with ResNet backbone):
- Level 0: stride ≈ 8x (from res3 or FPN output)
- Level 1: stride ≈ 16x (from res4 or FPN output)
- Level 2: stride ≈ 32x (from res5 or FPN output)

**For ViT backbone** (like in DVIS-DAQ):
- The pixel decoder receives ViT features which may have different strides
- Need to check the actual feature map dimensions during inference

## How Mask2Former Visualizes Cross-Attention (Per Paper Appendix)

**Important Clarification**: There is only **ONE type of attention mechanism** - cross-attention. The distinction between "cross-attention" and "masked attention" in the paper refers to whether the **mask constraint is applied** or not.

According to the Mask2Former paper appendix, they visualize attention from the **last three decoder layers** for a single query in two ways:

1. **Cross-Attention (Unmasked)** (Figure Ia top): Attention weights computed **without** the mask constraint (`memory_mask=None`)
2. **Masked Attention** (Figure Ia bottom): Attention weights computed **with** the mask constraint (`memory_mask=attn_mask`) - this is what actually happens during normal inference

### Understanding Masked Attention in DVIS-DAQ

In the DVIS-DAQ model (which uses Mask2Former's transformer decoder), the cross-attention mechanism works as follows:

1. **After each decoder layer**, a mask is predicted from the query features
2. This predicted mask is converted to an `attn_mask` that constrains attention
3. The `attn_mask` is passed as `memory_mask` to the **next layer's** cross-attention
4. This prevents the model from attending to regions where the predicted mask probability < 0.5

**Key code flow** (from `mask2former_transformer_decoder.py`):
```python
# After layer i, predict mask
outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(...)
# attn_mask shape: (B*num_heads, num_queries, H*W)
# True = don't attend, False = can attend

# Next layer (i+1) uses this mask
output = self.transformer_cross_attention_layers[i+1](
    output, src[level_index],
    memory_mask=attn_mask,  # <-- This constrains attention
    ...
)
```

**In normal inference**: You only get **masked attention** (with the constraint applied). This is what you'll extract when hooking into the forward pass.

### What to Visualize (Per Paper)

For each of the **last 3 decoder layers** (e.g., layers 7, 8, 9 for a 10-layer decoder):

1. **Masked Attention Visualization** (What you get from normal hooks):
   - Extract attention weights from the normal forward pass
   - These weights already have the mask constraint applied (`memory_mask=attn_mask`)
   - Regions where the mask probability < 0.5 will have zero or very low attention
   - Average across heads: `attn.mean(dim=1)`
   - Reshape to 2D: `(H_feat, W_feat)`
   - Upsample to image resolution using bilinear interpolation
   - Overlay on original image
   - **This is what the paper shows in Figure Ia bottom**

2. **Cross-Attention Visualization (Unmasked)** (Requires special extraction):
   - To see attention **without** the mask constraint, you'd need to:
     - Modify the forward pass to also run cross-attention with `memory_mask=None`
     - Or extract attention weights before the mask is applied (requires deeper hooking)
   - This shows how the query would attend if not constrained by the predicted mask
   - **This is what the paper shows in Figure Ia top** (for comparison)

### Key Implementation Details

- **Layers**: Last 3 layers (e.g., if decoder has 10 layers, visualize layers 7, 8, 9)
- **Query**: Single query that predicts the target object (e.g., "cat")
- **Heads**: Average across all attention heads
- **Upsampling**: Bilinear interpolation to image resolution
- **Visualization**: Overlay as heatmap on original image

### Practical Example (Mask2Former Paper Style)

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def visualize_mask2former_attention(attention_data_path, metadata_path, 
                                     image_path, query_idx, layer_idx, 
                                     attention_type='masked'):
    """
    Visualize attention following Mask2Former paper appendix.
    
    Args:
        attention_data_path: Path to .npz file with attention weights
        metadata_path: Path to .meta.json file with metadata
        image_path: Path to original image
        query_idx: Index of query to visualize (e.g., query that predicts "cat")
        layer_idx: Layer index (should be one of last 3 layers)
        attention_type: 'cross' (raw) or 'masked' (with mask constraint)
    """
    # Load data
    attention_data = np.load(attention_data_path)
    attention_weights = attention_data['attention_weights']  # (1, num_heads, num_queries, H*W)
    metadata = json.load(open(metadata_path))
    
    # Get dimensions
    H_feat = metadata['spatial_shape'][0]
    W_feat = metadata['spatial_shape'][1]
    
    # Load original image
    image = Image.open(image_path)
    original_height, original_width = image.size[1], image.size[0]
    
    # Extract attention for specific query
    # Shape: (batch, num_heads, num_queries, H*W)
    attn_per_head = attention_weights[0, :, query_idx, :]  # (num_heads, H*W)
    
    # Average across heads (as done in paper)
    attn_avg = attn_per_head.mean(axis=0)  # (H*W,)
    
    # Reshape to 2D feature map
    attn_2d = attn_avg.reshape(H_feat, W_feat)
    
    # Upsample to image resolution using bilinear interpolation
    scale_h = original_height / H_feat
    scale_w = original_width / W_feat
    attn_upsampled = zoom(attn_2d, (scale_h, scale_w), order=1)
    
    # Normalize for visualization
    attn_upsampled = (attn_upsampled - attn_upsampled.min()) / \
                     (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    im = ax.imshow(attn_upsampled, alpha=0.5, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Query {query_idx} {attention_type.capitalize()} Attention '
                 f'(Layer {layer_idx}, Averaged Across Heads)')
    plt.colorbar(im, ax=ax)
    return fig

# Example: Visualize last 3 layers for a query
num_decoder_layers = 10  # Adjust based on your model
last_3_layers = [num_decoder_layers - 3, num_decoder_layers - 2, num_decoder_layers - 1]
query_idx = 0  # Query that predicts your target object

for layer_idx in last_3_layers:
    # Masked attention (normal forward pass with mask constraint)
    fig_masked = visualize_mask2former_attention(
        f'video_12_frame_116_predictor_cross_attn_layer_{layer_idx}.npz',
        f'video_12_frame_116_predictor_cross_attn_layer_{layer_idx}.meta.json',
        'video_12_frame_116.jpg',
        query_idx, layer_idx, attention_type='masked'
    )
    fig_masked.savefig(f'query_{query_idx}_layer_{layer_idx}_masked_attention.png')
    
    # Note: To visualize cross-attention (without mask), you'd need to extract
    # attention weights with memory_mask=None, which requires modifying the forward pass
```

### Important Note: What You Actually Extract

**In DVIS-DAQ (and Mask2Former), there is only ONE attention mechanism: cross-attention.**

When you hook into `nn.MultiheadAttention` during the normal forward pass, you extract **masked attention** - this is cross-attention **with** the mask constraint applied (`memory_mask=attn_mask`). This is:
- What actually happens during inference
- What the paper shows in Figure Ia bottom
- What you'll get from normal hooks

**The "two types" in the paper are:**
- **Masked attention** (what you extract): Cross-attention WITH mask constraint (normal operation)
- **Unmasked attention** (for comparison): Cross-attention WITHOUT mask constraint (requires special extraction)

**For practical purposes**: You only need to extract masked attention (what you get from normal hooks). This shows how the model actually attends when constrained by the predicted mask, which is the key innovation of Mask2Former. The unmasked version is mainly shown in the paper for comparison to demonstrate the effect of the mask constraint.

### Key Differences

| Aspect | Our Detailed Approach | Mask2Former-Style |
|--------|----------------------|-------------------|
| **Mapping** | Feature location → image region (exact) | Bilinear upsampling (smooth) |
| **Heads** | Can visualize per-head or averaged | Typically averaged across heads |
| **Layers** | Any layer | Usually final/last few layers |
| **Resolution** | Exact pixel mapping | Smooth interpolation |
| **Use case** | Precise spatial analysis | General attention visualization |

## Key Differences from Backbone ViT Attention

| Aspect | Backbone ViT Attention | Pixel Decoder Cross-Attention |
|--------|------------------------|-------------------------------|
| **Spatial units** | 16×16 patches | Multi-scale feature locations |
| **Mapping** | Direct: `patch_idx → (row, col) → image_coords` | Indirect: `feat_idx → (h_feat, w_feat) → image_coords` |
| **Stride** | Fixed 16× (patch size) | Variable (8×, 16×, 32× depending on level) |
| **Dimensions** | `(H_patches, W_patches)` from padded image | `(H_feat, W_feat)` per feature level |
| **Query type** | Patch-to-patch (spatial) | Object query-to-spatial feature |

## Summary

To visualize pixel decoder cross-attention on video frames:

1. **Extract attention weights** with shape `(batch, heads, queries, H_feat*W_feat)`
2. **Capture spatial dimensions** `(H_feat, W_feat)` and feature level for each layer
3. **Determine feature stride** for each level (from config or by comparing dimensions)
4. **Reshape** attention to 2D grid: `(H_feat, W_feat)`
5. **Map** each feature location to image coordinates using stride
6. **Upsample/overlay** attention heatmap on original image

The attention map shows **which spatial regions in the pixel decoder features each object query is focusing on**, which corresponds to regions in the original image where the model is looking to detect/segment objects.

