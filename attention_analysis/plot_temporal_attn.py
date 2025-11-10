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
    
    # Compute column averages from the non-normalized rollout matrix
    col_avg = rollout.mean(axis=0)  # Average each column -> shape [num_frames]
    
    # Min-max normalize the column averages to [0, 1]
    col_min = col_avg.min()
    col_max = col_avg.max()
    col_range = col_max - col_min
    if col_range == 0:
        col_avg_norm = np.zeros_like(col_avg)  # All zeros if constant
    else:
        col_avg_norm = (col_avg - col_min) / col_range
    
    # Plot column averages as a line plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(T), col_avg_norm, linewidth=2)
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Normalized Column Average', fontsize=11)
    ax.set_title(f'Rollout Column Averages (Normalized)\nVideo {video_id}, Refiner ID {refiner_id}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    
    out_path_colavg = os.path.join(out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_colavg.png")
    plt.savefig(out_path_colavg, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved column averages plot to {out_path_colavg}")


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
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory for plots (default: <run_dir>/inference/attention_plots)")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "inference", "attention_plots")

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


