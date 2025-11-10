import os
import json
import argparse
import glob

import numpy as np


def load_results_temporal(run_dir):
    """Load results_temporal.json from inference directory."""
    inf_dir = os.path.join(run_dir, "inference")
    results_path = os.path.join(inf_dir, "results_temporal.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find results_temporal.json at {results_path}")
    with open(results_path, "r") as f:
        return json.load(f)


def pick_top_predictions_by_video(predictions):
    """Find top-scoring prediction for each video."""
    top_by_video = {}
    for pred in predictions:
        vid = pred.get("video_id")
        score = float(pred.get("score", -1.0))
        if vid is None:
            continue
        if vid not in top_by_video or score > top_by_video[vid]["score"]:
            top_by_video[vid] = pred
    return top_by_video


def load_attention_for_video(run_dir, video_id):
    """Load attention maps npz and metadata for a video.
    
    Looks for files matching pattern: video_{video_id}_*temporal_refiner_attn.npz
    in the attention_maps directory (not inference/attention_maps).
    """
    # Try both locations: attention_maps/ and inference/attention_maps/
    attn_dir1 = os.path.join(run_dir, "attention_maps")
    attn_dir2 = os.path.join(run_dir, "inference", "attention_maps")
    
    # Search for files matching the pattern
    npz_patterns = [
        os.path.join(attn_dir1, f"video_{video_id}_*temporal_refiner_attn.npz"),
        os.path.join(attn_dir2, f"video_{video_id}_*temporal_refiner_attn.npz"),
        os.path.join(attn_dir1, f"video_{video_id}.npz"),  # Fallback to simple pattern
        os.path.join(attn_dir2, f"video_{video_id}.npz"),  # Fallback to simple pattern
    ]
    
    npz_path = None
    for pattern in npz_patterns:
        matches = glob.glob(pattern)
        if matches:
            npz_path = matches[0]  # Use first match
            break
    
    if npz_path is None:
        raise FileNotFoundError(f"Could not find attention npz file for video {video_id} in {attn_dir1} or {attn_dir2}")
    
    # Find corresponding meta.json file
    meta_path = npz_path.replace(".npz", ".meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find metadata file for video {video_id}: {meta_path}")
    
    arrays = np.load(npz_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return arrays, meta


def select_instance_attention(attn_array: np.ndarray, refiner_id: int) -> np.ndarray:
    """Select instance-specific (T x T) attention from an array.
    
    Heuristics:
      - If 2D -> assume already (T, T)
      - If 3D -> try axis 0, else axis 1, pick slice by refiner_id
      - If 4D+ -> find last two dims with equal size (T, T), choose a leading axis with size > refiner_id
    """
    arr = attn_array
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Try indexing on axis 0 first
        if arr.shape[0] > refiner_id and arr.shape[-1] == arr.shape[-2]:
            return arr[refiner_id]
        # Try axis 1
        if arr.shape[1] > refiner_id and arr.shape[-1] == arr.shape[-2]:
            return arr[:, refiner_id, :]
        # Fallback: take first slice
        return arr[0] if arr.shape[0] > 0 else arr.squeeze()
    # For 4D or more
    if arr.ndim >= 4:
        T1, T2 = arr.shape[-2], arr.shape[-1]
        # Candidate instance axes are all dims before the last two
        leading_shape = arr.shape[:-2]
        # Prefer an axis with size > refiner_id
        for axis, size in enumerate(leading_shape):
            if size > refiner_id:
                slicer = [slice(None)] * arr.ndim
                slicer[axis] = refiner_id
                return arr[tuple(slicer)]
        # Fallback: squeeze leading dims
        return arr.reshape(-1, T1, T2)[0]
    # Fallback unknown
    return arr.squeeze()


def get_all_layers_sorted(meta):
    """Extract all layers from metadata, sorted by layer index if available."""
    layer_to_keys = {}
    for entry in meta:
        layer = entry.get("layer")
        key = entry.get("key")
        if layer and key:
            if layer not in layer_to_keys:
                layer_to_keys[layer] = []
            layer_to_keys[layer].append(key)
    
    # Sort layers by natural order in name if possible
    def layer_sort_key(name):
        import re
        # Match patterns like "self_attention_layers.0" or "transformer_time_self_attention_layers.0"
        m = re.search(r"(?:transformer_time_)?self_attention_layers\.(\d+)", name)
        return int(m.group(1)) if m else 1e9
    
    sorted_layers = sorted(layer_to_keys.keys(), key=layer_sort_key)
    return sorted_layers, layer_to_keys


def normalize_rows(matrix):
    """Row normalize a matrix so each row sums to 1.0."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
    return matrix / row_sums


def compute_rollout(run_dir, video_id, refiner_id):
    """Compute attention rollout: T_6 * T_5 * ... * T_0 where T_i = normalize_rows(A_i + I)."""
    arrays, meta = load_attention_for_video(run_dir, video_id)
    sorted_layers, layer_to_keys = get_all_layers_sorted(meta)
    
    if not sorted_layers:
        raise ValueError(f"No attention layers found for video {video_id}")
    
    # Collect attention matrices for each layer (using last occurrence per layer)
    attention_matrices = []
    layer_names = []
    
    for layer in sorted_layers:
        keys = layer_to_keys[layer]
        # Use the last key for this layer
        key = keys[-1]
        if key not in arrays:
            continue
        arr = arrays[key]
        attn_mat = select_instance_attention(arr, refiner_id)
        
        # Ensure 2D
        attn_mat = np.array(attn_mat)
        if attn_mat.ndim != 2:
            attn_mat = np.squeeze(attn_mat)
            if attn_mat.ndim != 2:
                continue
        
        attention_matrices.append(attn_mat)
        layer_names.append(layer)
    
    if not attention_matrices:
        raise ValueError(f"No valid attention matrices found for video {video_id}, refiner_id {refiner_id}")
    
    # Ensure all matrices have the same shape
    shapes = [m.shape for m in attention_matrices]
    if len(set(shapes)) > 1:
        print(f"Warning: Attention matrices have different shapes: {shapes}")
        # Use the most common shape
        from collections import Counter
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        # Resize all to most common shape if possible
        resized = []
        for m in attention_matrices:
            if m.shape != most_common_shape:
                # Simple interpolation if needed
                from scipy.ndimage import zoom
                scale = (most_common_shape[0] / m.shape[0], most_common_shape[1] / m.shape[1])
                m = zoom(m, scale, order=1)
            resized.append(m)
        attention_matrices = resized
    
    T, _ = attention_matrices[0].shape
    
    # Compute T_i = normalize_rows(A_i + I) for each layer
    identity = np.eye(T)
    transformed = []
    for i, A_i in enumerate(attention_matrices):
        T_i = A_i + identity
        T_i = normalize_rows(T_i)  # Row normalize after adding identity
        transformed.append(T_i)
    
    # Multiply matrices in order: T_6 * T_5 * ... * T_0 (or last * ... * first)
    # Start from the last layer (highest index) and multiply backward
    rollout = transformed[-1]  # Start with T_N (last layer)
    for i in range(len(transformed) - 2, -1, -1):  # Go backward from T_{N-1} to T_0
        rollout = np.dot(rollout, transformed[i])
    
    return rollout, layer_names, attention_matrices, transformed


def main():
    parser = argparse.ArgumentParser(description="Compute temporal attention rollout")
    parser.add_argument("run_dir", help="Evaluation run directory (contains inference/results_temporal.json)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Optional override for base output directory (default: <run_dir>)")
    args = parser.parse_args()
    
    run_dir = args.run_dir
    base_dir = args.out_dir or run_dir
    # Save to attention_maps/rolled_out/ (not inference/attention_maps/)
    attn_maps_dir = os.path.join(base_dir, "attention_maps")
    out_maps_dir = os.path.join(attn_maps_dir, "rolled_out")
    os.makedirs(out_maps_dir, exist_ok=True)
    
    predictions = load_results_temporal(run_dir)
    top_by_video = pick_top_predictions_by_video(predictions)
    
    for vid, pred in top_by_video.items():
        if "refiner_id" not in pred:
            print(f"Skipping video {vid}: refiner_id missing in results_temporal.json entry")
            continue
        refiner_id = int(pred["refiner_id"])
        try:
            rollout, layer_names, attention_matrices, transformed = compute_rollout(run_dir, vid, refiner_id)
            
            # Save rollout matrix as npz
            rollout_path = os.path.join(out_maps_dir, f"video_{vid}_refiner_{refiner_id}_rollout.npz")
            np.savez_compressed(rollout_path, rollout=rollout)
            print(f"Saved rollout matrix to {rollout_path}")
        except FileNotFoundError as e:
            print(str(e))
        except Exception as e:
            print(f"Failed rollout for video {vid}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

