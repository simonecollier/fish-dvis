import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


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
    """Load attention maps npz and metadata for a video."""
    inf_dir = os.path.join(run_dir, "inference")
    attn_dir = os.path.join(inf_dir, "attention_maps")
    npz_path = os.path.join(attn_dir, f"video_{video_id}.npz")
    meta_path = os.path.join(attn_dir, f"video_{video_id}.meta.json")
    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing attention files for video {video_id}: {npz_path}, {meta_path}")
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
        m = re.search(r"self_attention_layers\.(\d+)", name)
        return int(m.group(1)) if m else 1e9
    
    sorted_layers = sorted(layer_to_keys.keys(), key=layer_sort_key)
    return sorted_layers, layer_to_keys


def compute_rollout(run_dir, video_id, refiner_id):
    """Compute attention rollout: T_6 * T_5 * ... * T_0 where T_i = A_i + I."""
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
    
    # Compute T_i = A_i + I for each layer
    identity = np.eye(T)
    transformed = []
    for i, A_i in enumerate(attention_matrices):
        T_i = A_i + identity
        transformed.append(T_i)
    
    # Multiply matrices in order: T_6 * T_5 * ... * T_0 (or last * ... * first)
    # Start from the last layer (highest index) and multiply backward
    rollout = transformed[-1]  # Start with T_N (last layer)
    for i in range(len(transformed) - 2, -1, -1):  # Go backward from T_{N-1} to T_0
        rollout = np.dot(rollout, transformed[i])
    
    return rollout, layer_names, attention_matrices, transformed


def plot_rollout(run_dir, video_id, refiner_id, rollout, layer_names, attention_matrices, transformed, out_plot_dir, out_maps_dir):
    """Plot the rollout result and save matrices to attention_maps."""
    T, _ = rollout.shape
    
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_maps_dir, exist_ok=True)
    
    # Plot rollout result
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(rollout, cmap='viridis', aspect='equal')
    ax.set_title(f'Rollout (T_6 * ... * T_0)\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    ax.set_box_aspect(1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    out_path = os.path.join(out_plot_dir, f"video_{video_id}_refiner_{refiner_id}_rollout.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved rollout plot to {out_path}")

    # Row-normalized rollout (each row min-max scaled to [0,1])
    rmin = rollout.min(axis=1, keepdims=True)
    rmax = rollout.max(axis=1, keepdims=True)
    denom = np.where((rmax - rmin) == 0, 1.0, (rmax - rmin))
    rollout_norm = (rollout - rmin) / denom

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(rollout_norm, cmap='viridis', aspect='equal')
    ax.set_title(f'Rollout Row-Normalized\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    ax.set_box_aspect(1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path_norm = os.path.join(out_plot_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm.png")
    plt.savefig(out_path_norm, dpi=150, bbox_inches='tight')
    plt.close()

    # Save normalized matrix
    rollout_path_norm = os.path.join(out_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm.npz")
    np.savez_compressed(rollout_path_norm, rollout_rownorm=rollout_norm)
    print(f"Saved row-normalized rollout plot to {out_path_norm}")

    # Save column averages of the row-normalized rollout (mean over rows for each column)
    col_avg = rollout_norm.mean(axis=0)
    colavg_npz_path = os.path.join(out_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm_colavg.npz")
    np.savez_compressed(colavg_npz_path, rollout_rownorm_colavg=col_avg)
    # Also save as CSV for convenience
    colavg_csv_path = os.path.join(out_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm_colavg.csv")
    try:
        import csv
        with open(colavg_csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["column_index", "mean_value"])
            for idx, val in enumerate(col_avg):
                writer.writerow([idx, float(val)])
    except Exception:
        pass

    # Min-max normalize the column averages to [0, 1]
    cav_min = float(col_avg.min()) if col_avg.size > 0 else 0.0
    cav_max = float(col_avg.max()) if col_avg.size > 0 else 1.0
    cav_denom = (cav_max - cav_min) if (cav_max - cav_min) != 0 else 1.0
    col_avg_norm = (col_avg - cav_min) / cav_denom

    colavg_norm_npz_path = os.path.join(out_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm_colavg_norm.npz")
    np.savez_compressed(colavg_norm_npz_path, rollout_rownorm_colavg_norm=col_avg_norm)

    colavg_norm_csv_path = os.path.join(out_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm_colavg_norm.csv")
    try:
        import csv
        with open(colavg_norm_csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(["column_index", "mean_value_norm"])
            for idx, val in enumerate(col_avg_norm):
                writer.writerow([idx, float(val)])
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Compute and plot temporal attention rollout")
    parser.add_argument("run_dir", help="Evaluation run directory (contains inference/results_temporal.json)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Optional override for base output directory (default: <run_dir>)")
    args = parser.parse_args()
    
    run_dir = args.run_dir
    base_dir = args.out_dir or run_dir
    # Per request: plots -> inference/attention_plots, matrices -> inference/attention_maps
    out_plot_dir = os.path.join(base_dir, "inference", "attention_plots")
    out_maps_dir = os.path.join(base_dir, "inference", "attention_maps")
    
    predictions = load_results_temporal(run_dir)
    top_by_video = pick_top_predictions_by_video(predictions)
    
    for vid, pred in top_by_video.items():
        if "refiner_id" not in pred:
            print(f"Skipping video {vid}: refiner_id missing in results_temporal.json entry")
            continue
        refiner_id = int(pred["refiner_id"])
        try:
            rollout, layer_names, attention_matrices, transformed = compute_rollout(run_dir, vid, refiner_id)
            plot_rollout(run_dir, vid, refiner_id, rollout, layer_names, attention_matrices, transformed, out_plot_dir, out_maps_dir)
            
            # Also save rollout matrix as npz
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

