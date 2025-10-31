import os
import json
import argparse
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


def plot_attention_grid_for_video(run_dir, video_id, refiner_id, out_dir):
    arrays, meta = load_attention_for_video(run_dir, video_id)
    layer_to_key = pick_last_per_layer(meta)
    if not layer_to_key:
        print(f"No attention metadata for video {video_id}")
        return

    # Sort layers by natural order in name if possible
    def layer_sort_key(name):
        # Expect names like '...transformer_time_self_attention_layers.X...' -> pull X if present
        import re
        m = re.search(r"self_attention_layers\.(\d+)", name)
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
            plot_attention_grid_for_video(run_dir, vid, refiner_id, out_dir)
        except FileNotFoundError as e:
            print(str(e))
        except Exception as e:
            print(f"Failed plotting for video {vid}: {e}")


if __name__ == "__main__":
    main()


