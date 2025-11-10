# Attention Rollout Computation Process

This document describes the exact step-by-step process used to compute rolled out attention maps for both **spatial** and **temporal** attention.

---

## Spatial Rollout

Spatial rollout computes attention across spatial patches within a single frame, rolling out across all 24 backbone ViT layers.

### Overview

For each frame, we take attention tensors from all 24 layers (with multiple attention heads), average across heads, process them to simulate residual connections, and multiply them together in sequence to get the final rolled out attention map.

### Step-by-Step Process

#### Input Data

For a single frame, we have:
- 24 attention tensors: `A₀, A₁, A₂, ..., A₂₃`
- Each tensor has shape `[num_heads, 433, 433]` (num_heads × num_patches × num_patches)
- Each tensor `Aᵢ` represents the attention weights from layer `i` across all attention heads

#### Step 0: Average Across Attention Heads

For each layer `i` (from 0 to 23), we first average across the attention heads dimension:

```
Aᵢ_avg = mean(Aᵢ, axis=0)  # Average across heads dimension (axis 0 for single frame)
```

This reduces the shape from `[num_heads, 433, 433]` to `[433, 433]`.

**Note**: In the actual implementation, all frames are processed together, so the input shape is `[num_frames, num_heads, 433, 433]` and averaging is done along `axis=1` (the heads dimension). For a single frame, we extract one frame first, giving `[num_heads, 433, 433]`, then average along `axis=0`.

**Result**: We now have 24 averaged attention matrices: `A₀_avg, A₁_avg, A₂_avg, ..., A₂₃_avg`, each with shape `[433, 433]`

#### Step 1: Process Each Layer (Simulate Residual Connection)

For each layer `i` (from 0 to 23), we transform the averaged attention matrix `Aᵢ_avg`:

##### 1.1 Add Identity Matrix
```
Ãᵢ = Aᵢ_avg + I
```
where `I` is the identity matrix of shape `[433, 433]`.

This simulates the residual connection in the transformer architecture.

##### 1.2 Row Normalize
```
row_sums = sum(Ãᵢ, axis=1)  # Sum each row
Āᵢ = Ãᵢ / row_sums  # Divide each row by its sum
```

This ensures each row of `Āᵢ` sums to 1.0, making it a proper probability distribution.

**Result**: We now have 24 processed matrices: `Ā₀, Ā₁, Ā₂, ..., Ā₂₃`

#### Step 2: Multiply Matrices in Sequence

We start with the identity matrix and multiply the processed attention matrices in order:

##### 2.1 Initialize
```
R = I  # Identity matrix [433, 433]
```

##### 2.2 Iterative Multiplication

For each layer `i` from 0 to 23:
```
R = Āᵢ @ R
```

This builds up the product step by step:
- After layer 0: `R = Ā₀ @ I = Ā₀`
- After layer 1: `R = Ā₁ @ Ā₀`
- After layer 2: `R = Ā₂ @ Ā₁ @ Ā₀`
- ...
- After layer 23: `R = Ā₂₃ @ Ā₂₂ @ ... @ Ā₁ @ Ā₀`

**Mathematical expression**:
```
R = Ā₂₃ @ Ā₂₂ @ ... @ Ā₂ @ Ā₁ @ Ā₀
```

#### Step 3: Result

The final matrix `R` is the rolled out attention map for this frame, with shape `[433, 433]`.

**Note**: No final row normalization is applied after matrix multiplication. The result is the product of the normalized layer matrices.

### Complete Formula (Spatial)

Putting it all together for a single frame:

1. For each layer `i` (0 to 23), average across heads:
   ```
   Aᵢ_avg = mean(Aᵢ, axis=0)
   ```

2. For each layer `i` (0 to 23), simulate residual connection:
   ```
   Āᵢ = normalize_rows(Aᵢ_avg + I)
   ```

3. Multiply in sequence:
   ```
   R = Ā₂₃ @ Ā₂₂ @ ... @ Ā₂ @ Ā₁ @ Ā₀
   ```

**Note**: No final row normalization is applied. The result is the product of the normalized layer matrices.

### Implementation Details (Spatial)

#### Head Averaging Function

```python
def collapse_heads_in_memory(attn_weights):
    # attn_weights shape: [num_frames, num_heads, num_patches, num_patches]
    # Returns shape: [num_frames, num_patches, num_patches]
    return np.mean(attn_weights, axis=1)  # Average across heads dimension
```

#### Row Normalization Function

```python
def normalize_rows(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
    return matrix / row_sums
```

#### Matrix Multiplication Order

The multiplication is done **right-to-left** in terms of layer indices:
- Start with layer 0 (rightmost)
- Multiply by layer 1, then 2, ..., up to layer 23 (leftmost)
- This gives: `Ā₂₃ @ Ā₂₂ @ ... @ Ā₁ @ Ā₀`

This order is correct because in matrix multiplication, the rightmost matrix is applied first to the input, and subsequent matrices are applied to the result.

---

## Temporal Rollout

Temporal rollout computes attention across frames (temporal dimension), rolling out across all temporal attention layers in the refiner module.

### Overview

For a specific video instance (identified by `refiner_id`), we take attention matrices from all temporal attention layers, add identity matrices to simulate residual connections, and multiply them together in sequence to get the final rolled out attention map across frames.

### Step-by-Step Process

#### Input Data

For a video instance, we have:
- Multiple attention matrices: `A₀, A₁, A₂, ..., A_N` (where N is the number of temporal attention layers, typically 6-7)
- Each matrix has shape `[T, T]` where `T` is the number of frames
- Each matrix `Aᵢ` represents the temporal attention weights from layer `i` for a specific instance (refiner_id)
- The matrices are extracted from `self_attention_layers` in the refiner module

#### Step 0: Select Instance-Specific Attention

For each layer `i`, we extract the instance-specific attention matrix:

```
Aᵢ = select_instance_attention(attn_array, refiner_id)
```

This selects the `[T, T]` attention matrix corresponding to the specific instance (refiner_id) from the attention array. The selection handles different array shapes (2D, 3D, 4D+) by finding the appropriate slice.

**Result**: We have N attention matrices: `A₀, A₁, A₂, ..., A_N`, each with shape `[T, T]`

#### Step 1: Process Each Layer (Simulate Residual Connection)

For each layer `i` (from 0 to N), we transform the attention matrix `Aᵢ`:

##### 1.1 Add Identity Matrix
```
T̃ᵢ = Aᵢ + I
```
where `I` is the identity matrix of shape `[T, T]`.

This simulates the residual connection in the transformer architecture.

##### 1.2 Row Normalize
```
row_sums = sum(T̃ᵢ, axis=1)  # Sum each row
Tᵢ = T̃ᵢ / row_sums  # Divide each row by its sum
```

This ensures each row of `Tᵢ` sums to 1.0, making it a proper probability distribution.

**Result**: We now have N transformed matrices: `T₀, T₁, T₂, ..., T_N`

#### Step 2: Multiply Matrices in Sequence

We start with the last layer and multiply backward through all layers:

##### 2.1 Initialize
```
rollout = T_N  # Start with the last layer (highest index)
```

##### 2.2 Iterative Multiplication

For each layer `i` from N-1 down to 0:
```
rollout = rollout @ Tᵢ
```

This builds up the product step by step:
- Start: `rollout = T_N`
- After layer N-1: `rollout = T_N @ T_{N-1}`
- After layer N-2: `rollout = T_N @ T_{N-1} @ T_{N-2}`
- ...
- After layer 0: `rollout = T_N @ T_{N-1} @ ... @ T₁ @ T₀`

**Mathematical expression**:
```
rollout = T_N @ T_{N-1} @ ... @ T₂ @ T₁ @ T₀
```

**Note**: Unlike spatial rollout, temporal rollout multiplies from the **last layer to the first** (left-to-right in terms of layer indices).

#### Step 3: Result

The final matrix `rollout` is the rolled out attention map across frames, with shape `[T, T]`.

**Note**: No final row normalization is applied after matrix multiplication. The result is the product of the normalized layer matrices.

### Complete Formula (Temporal)

Putting it all together:

1. For each layer `i` (0 to N), select instance-specific attention:
   ```
   Aᵢ = select_instance_attention(attn_array, refiner_id)
   ```

2. For each layer `i` (0 to N), simulate residual connection and normalize:
   ```
   Tᵢ = normalize_rows(Aᵢ + I)
   ```

3. Multiply in sequence (from last to first):
   ```
   rollout = T_N @ T_{N-1} @ ... @ T₁ @ T₀
   ```

**Note**: No final row normalization is applied. The result is the product of the normalized layer matrices.

### Implementation Details (Temporal)

#### Row Normalization Function

```python
def normalize_rows(matrix):
    """Row normalize a matrix so each row sums to 1.0."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
    return matrix / row_sums
```

#### Instance Selection Function

```python
def select_instance_attention(attn_array, refiner_id):
    """Select instance-specific (T x T) attention from an array.
    
    Handles different array shapes:
    - 2D: already (T, T) -> return as-is
    - 3D: try axis 0 or 1, pick slice by refiner_id
    - 4D+: find last two dims with equal size (T, T), choose leading axis with size > refiner_id
    """
    # Implementation handles various shapes and extracts the correct slice
    return attn_matrix  # Shape [T, T]
```

#### Matrix Multiplication Order

The multiplication is done **left-to-right** in terms of layer indices:
- Start with the last layer (highest index, N)
- Multiply by layer N-1, then N-2, ..., down to layer 0 (first layer)
- This gives: `T_N @ T_{N-1} @ ... @ T₁ @ T₀`

### Post-Processing (Temporal)

**Note**: The temporal rollout script (`temporal_rollout.py`) only computes and saves the rollout matrix. All plotting and visualization is handled by a separate script (`plot_temporal_attn.py`).

The plotting script applies additional processing for visualization:

1. **Min-max normalized rollout** (for visualization):
   ```
   rmin = rollout.min(axis=1, keepdims=True)
   rmax = rollout.max(axis=1, keepdims=True)
   rollout_minmax = (rollout - rmin) / (rmax - rmin)  # Min-max normalize each row to [0,1]
   ```
   Note: This is different from row normalization. Row normalization makes rows sum to 1, while min-max normalization scales each row to [0,1] range.

2. **Plotting**: The plotting script generates visualizations of both the raw rollout and the min-max normalized version.

---

## Key Differences Between Spatial and Temporal Rollout

| Aspect | Spatial Rollout | Temporal Rollout |
|--------|----------------|------------------|
| **Input dimensions** | `[num_heads, num_patches, num_patches]` | `[T, T]` (already 2D per instance) |
| **Head averaging** | Yes (averages across heads) | No (already per-instance) |
| **Row normalization after identity** | Yes | Yes |
| **Multiplication order** | Layer 0 → Layer 23 (right-to-left) | Layer N → Layer 0 (left-to-right) |
| **Final row normalization** | No | No |
| **Output** | Product of normalized matrices | Product of normalized matrices |
| **Number of layers** | 24 (backbone ViT layers) | N (typically 6-7, refiner temporal layers) |
| **Matrix size** | 433 × 433 (patches) | T × T (frames) |
| **Plotting** | Separate script | Separate script (`plot_temporal_attn.py`) |
