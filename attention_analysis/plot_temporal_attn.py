import os
import json
import argparse
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results_temporal(run_dir):
    inf_dir = os.path.join(run_dir, "inference")
    results_path = os.path.join(inf_dir, "results_temporal.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find results_temporal.json at {results_path}")
    with open(results_path, "r") as f:
        return json.load(f)


def pick_top_predictions_by_video(predictions):
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
    
    Supports multiple naming conventions:
    1. New: video_{video_id}_frames*-*_refiner_temporal_layer_*_attn.npz (multiple files per video)
    2. Old: video_{video_id}_*temporal_refiner_attn.npz (single consolidated file)
    3. Fallback: video_{video_id}.npz
    
    Only loads refiner attention maps (ignores predictor maps).
    """
    # Try both locations: attention_maps/ and inference/attention_maps/
    attn_dir1 = os.path.join(run_dir, "attention_maps")
    attn_dir2 = os.path.join(run_dir, "inference", "attention_maps")
    
    # First, try to find new format: multiple refiner temporal files per video
    # Pattern: video_{video_id}_*refiner_temporal*.npz (but NOT predictor)
    new_patterns = [
        os.path.join(attn_dir1, f"video_{video_id}_*refiner_temporal*.npz"),
        os.path.join(attn_dir2, f"video_{video_id}_*refiner_temporal*.npz"),
    ]
    
    npz_files = []
    for pattern in new_patterns:
        matches = glob.glob(pattern)
        # Filter out predictor files if any match
        matches = [m for m in matches if 'predictor' not in m.lower()]
        npz_files.extend(matches)
    
    if npz_files:
        # Load all refiner temporal files and combine them
        arrays = {}
        meta = []
        
        for npz_path in sorted(npz_files):
            # Load the npz file
            file_arrays = np.load(npz_path)
            
            # Find corresponding meta.json file
            meta_path = npz_path.replace(".npz", ".meta.json")
            if not os.path.exists(meta_path):
                print(f"Warning: Could not find metadata file for {npz_path}, skipping")
                continue
            
            with open(meta_path, "r") as f:
                file_meta = json.load(f)
            
            # Extract layer name from metadata
            layer = file_meta.get("layer")
            if not layer:
                # Try to infer from filename
                import re
                match = re.search(r'layer_(\d+)_attn', os.path.basename(npz_path))
                if match:
                    layer_idx = match.group(1)
                    layer = f"refiner.transformer_time_self_attention_layers.{layer_idx}"
                else:
                    layer = os.path.basename(npz_path).replace(".npz", "")
            
            # Use layer name as key, or generate a unique key
            # If multiple files have the same layer (different frame ranges), we'll use the last one
            key = layer
            arrays[key] = file_arrays['attention_weights'] if 'attention_weights' in file_arrays else list(file_arrays.values())[0]
            
            # Add metadata entry
            meta_entry = file_meta.copy()
            meta_entry['key'] = key
            meta.append(meta_entry)
        
        if arrays:
            return arrays, meta
    
    # Fallback to old format: single consolidated file
    old_patterns = [
        os.path.join(attn_dir1, f"video_{video_id}_*temporal_refiner_attn.npz"),
        os.path.join(attn_dir2, f"video_{video_id}_*temporal_refiner_attn.npz"),
        os.path.join(attn_dir1, f"video_{video_id}.npz"),  # Fallback to simple pattern
        os.path.join(attn_dir2, f"video_{video_id}.npz"),  # Fallback to simple pattern
    ]
    
    npz_path = None
    for pattern in old_patterns:
        matches = glob.glob(pattern)
        # Filter out predictor files if any match
        matches = [m for m in matches if 'predictor' not in m.lower()]
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
    """Best-effort selection of instance-specific (T x T) attention from an array.
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
    # Identify frame dims: choose last two equal dims as (T, T)
    if arr.ndim >= 4:
        T1, T2 = arr.shape[-2], arr.shape[-1]
        if T1 != T2:
            # If not square, try to find any pair of equal dims near the end
            # Fallback to squeeze to last two dims
            pass
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


def pick_last_per_layer(meta):
    """From metadata entries (ordered), pick the last key per layer name."""
    last_key_for_layer = {}
    for entry in meta:
        layer = entry.get("layer")
        key = entry.get("key")
        if layer and key:
            last_key_for_layer[layer] = key
    return last_key_for_layer


def normalize_rows(matrix):
    """Row normalize a matrix so each row sums to 1.0."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
    return matrix / row_sums


def load_rollout_for_video(run_dir, video_id, refiner_id):
    """Load rollout matrix for a video."""
    # Look in attention_maps/rolled_out/ (not inference/attention_maps/)
    attn_dir = os.path.join(run_dir, "attention_maps")
    rolled_out_dir = os.path.join(attn_dir, "rolled_out")
    rollout_path = os.path.join(rolled_out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout.npz")
    if not os.path.exists(rollout_path):
        raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
    data = np.load(rollout_path)
    return data['rollout']


def load_col_avg_norm_for_video(run_dir, video_id, refiner_id):
    """Load normalized column averages for a video (pre-computed by temporal_rollout.py)."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    rolled_out_dir = os.path.join(attn_dir, "rolled_out")
    col_avg_norm_path = os.path.join(rolled_out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_col_avg_norm.npz")
    if not os.path.exists(col_avg_norm_path):
        raise FileNotFoundError(f"Column averages file not found: {col_avg_norm_path}")
    data = np.load(col_avg_norm_path)
    return data['col_avg_norm']


def load_rollout_nolayer0_for_video(run_dir, video_id, refiner_id):
    """Load rollout matrix without layer 0 for a video."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    rolled_out_dir = os.path.join(attn_dir, "rolled_out")
    rollout_path = os.path.join(rolled_out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_nolayer0.npz")
    if not os.path.exists(rollout_path):
        raise FileNotFoundError(f"Rollout (no layer 0) file not found: {rollout_path}")
    data = np.load(rollout_path)
    return data['rollout']


def load_col_avg_norm_nolayer0_for_video(run_dir, video_id, refiner_id):
    """Load normalized column averages without layer 0 for a video (pre-computed by temporal_rollout.py)."""
    attn_dir = os.path.join(run_dir, "attention_maps")
    rolled_out_dir = os.path.join(attn_dir, "rolled_out")
    col_avg_norm_path = os.path.join(rolled_out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_nolayer0_col_avg_norm.npz")
    if not os.path.exists(col_avg_norm_path):
        raise FileNotFoundError(f"Column averages (no layer 0) file not found: {col_avg_norm_path}")
    data = np.load(col_avg_norm_path)
    return data['col_avg_norm']


def plot_rollout_for_video(run_dir, video_id, refiner_id, out_dir):
    """Plot the rollout matrix for a video."""
    try:
        rollout = load_rollout_for_video(run_dir, video_id, refiner_id)
    except FileNotFoundError:
        print(f"Rollout not found for video {video_id}, refiner {refiner_id}")
        return
    
    T, _ = rollout.shape
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Row-normalize the rollout matrix before plotting
    rollout_normalized = normalize_rows(rollout)
    
    # Plot row-normalized rollout result
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(rollout_normalized, cmap='viridis', aspect='equal')
    ax.set_title(f'Rollout (T_6 * ... * T_0) Row-Normalized\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    ax.set_box_aspect(1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved row-normalized rollout plot to {out_path}")
    
    # Load pre-computed normalized column averages (from temporal_rollout.py)
    try:
        col_avg_norm = load_col_avg_norm_for_video(run_dir, video_id, refiner_id)
    except FileNotFoundError:
        print(f"Warning: Normalized column averages not found for video {video_id}, refiner {refiner_id}")
        print(f"  Skipping column averages plot. Run temporal_rollout.py first.")
        return
    
    # Plot column averages as a line plot
    num_frames = len(col_avg_norm)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(num_frames), col_avg_norm, linewidth=2)
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Normalized Column Average', fontsize=11)
    ax.set_title(f'Rollout Column Averages (Normalized)\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_frames - 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    
    out_path_colavg = os.path.join(out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_colavg.png")
    plt.savefig(out_path_colavg, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved column averages plot to {out_path_colavg}")
    
    # Plot rollout without layer 0 (if available)
    try:
        rollout_nolayer0 = load_rollout_nolayer0_for_video(run_dir, video_id, refiner_id)
    except FileNotFoundError:
        print(f"Rollout (no layer 0) not found for video {video_id}, refiner {refiner_id}, skipping")
        return
    
    T_nolayer0, _ = rollout_nolayer0.shape
    
    # Row-normalize the rollout matrix before plotting
    rollout_nolayer0_normalized = normalize_rows(rollout_nolayer0)
    
    # Plot row-normalized rollout result (no layer 0)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(rollout_nolayer0_normalized, cmap='viridis', aspect='equal')
    ax.set_title(f'Rollout (T_6 * ... * T_1) Row-Normalized\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    ax.set_box_aspect(1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    out_path_nolayer0 = os.path.join(out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_nolayer0.png")
    plt.savefig(out_path_nolayer0, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved row-normalized rollout plot (no layer 0) to {out_path_nolayer0}")
    
    # Load pre-computed normalized column averages without layer 0
    try:
        col_avg_norm_nolayer0 = load_col_avg_norm_nolayer0_for_video(run_dir, video_id, refiner_id)
    except FileNotFoundError:
        print(f"Warning: Normalized column averages (no layer 0) not found for video {video_id}, refiner {refiner_id}")
        print(f"  Skipping column averages plot (no layer 0). Run temporal_rollout.py first.")
        return
    
    # Plot column averages as a line plot (no layer 0)
    num_frames_nolayer0 = len(col_avg_norm_nolayer0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(num_frames_nolayer0), col_avg_norm_nolayer0, linewidth=2)
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Normalized Column Average', fontsize=11)
    ax.set_title(f'Rollout Column Averages (Normalized, No Layer 0)\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_frames_nolayer0 - 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    
    out_path_colavg_nolayer0 = os.path.join(out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_nolayer0_colavg.png")
    plt.savefig(out_path_colavg_nolayer0, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved column averages plot (no layer 0) to {out_path_colavg_nolayer0}")


def plot_attention_grid_for_video(run_dir, video_id, refiner_id, out_dir):
    arrays, meta = load_attention_for_video(run_dir, video_id)
    layer_to_key = pick_last_per_layer(meta)
    if not layer_to_key:
        print(f"No attention metadata for video {video_id}")
        return

    # Sort layers by natural order in name if possible
    def layer_sort_key(name):
        # Match patterns like "self_attention_layers.0" or "transformer_time_self_attention_layers.0"
        import re
        m = re.search(r"(?:transformer_time_)?self_attention_layers\.(\d+)", name)
        return int(m.group(1)) if m else 1e9

    sorted_layers = sorted(layer_to_key.keys(), key=layer_sort_key)
    # Limit to 6 layers if more
    sorted_layers = sorted_layers[:6]

    attn_mats = []
    titles = []
    for layer in sorted_layers:
        key = layer_to_key[layer]
        if key not in arrays:
            continue
        arr = arrays[key]
        mat = select_instance_attention(arr, refiner_id)
        # Ensure 2D
        mat2d = np.array(mat)
        if mat2d.ndim != 2:
            # Best-effort squeeze
            mat2d = np.squeeze(mat2d)
            if mat2d.ndim != 2:
                # Cannot plot non-2D; skip
                continue
        attn_mats.append(mat2d)
        titles.append(os.path.basename(layer))

    if not attn_mats:
        print(f"No plottable attention matrices for video {video_id}")
        return

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    axes = axes.flatten()
    for i in range(min(len(attn_mats), rows * cols)):
        ax = axes[i]
        im = ax.imshow(attn_mats[i], cmap='viridis', aspect='equal')
        ax.set_title(titles[i], fontsize=9)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frame')
        # Ensure the axes box is square
        try:
            ax.set_box_aspect(1)
        except Exception:
            ax.set_aspect('equal', adjustable='box')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"video_{video_id}_refiner_{refiner_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved attention grid to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot temporal attention heatmaps per video")
    parser.add_argument("run_dir", help="Evaluation run directory (contains inference/results_temporal.json)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory for plots (default: <run_dir>/inference/temporal_attention_plots)")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "inference", "temporal_attention_plots")

    predictions = load_results_temporal(run_dir)
    top_by_video = pick_top_predictions_by_video(predictions)

    for vid, pred in top_by_video.items():
        if "refiner_id" not in pred:
            print(f"Skipping video {vid}: refiner_id missing in results_temporal.json entry")
            continue
        refiner_id = int(pred["refiner_id"])
        try:
            # Plot raw attention grids
            plot_attention_grid_for_video(run_dir, vid, refiner_id, out_dir)
            # Plot rollout
            plot_rollout_for_video(run_dir, vid, refiner_id, out_dir)
        except FileNotFoundError as e:
            print(str(e))
        except Exception as e:
            print(f"Failed plotting for video {vid}: {e}")


if __name__ == "__main__":
    main()


