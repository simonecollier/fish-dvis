import os
import json
import subprocess
from typing import Dict, Any, List, Tuple

import numpy as np

try:
    from pycocotools import mask as coco_mask
except Exception:  # pragma: no cover
    coco_mask = None


MODEL_DIR = "/home/simone/store/simone/dvis-model-outputs/trained_models/model_camera_s1_fixed"
VAL_JSON = os.path.join(MODEL_DIR, "val.json")
RESULTS_JSON = os.path.join(MODEL_DIR, "checkpoint_evaluations/checkpoint_0003635/inference/results.json")
ATTN_OUT_DIR = "/home/simone/store/simone/attention_maps_newtest"
ATTN_RUNNER = "/home/simone/fish-dvis/attention_analysis/run_attention_extraction_single_video.sh"
SKIP_RUN = True  # Do not extract again; use existing outputs only


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def ensure_attention_outputs(video_id: int) -> str:
    """
    Run attention extraction for a single video if the top prediction file is missing.
    Returns path to top prediction json.
    """
    top_pred_path = os.path.join(ATTN_OUT_DIR, f"top_prediction_video_{video_id}_top_1_rollout.json")
    if not os.path.exists(top_pred_path) and not SKIP_RUN:
        subprocess.run([ATTN_RUNNER, str(video_id)], check=True)
    return top_pred_path


def decode_rle(rle_obj: Any) -> np.ndarray:
    if coco_mask is None:
        return None  # pycocotools unavailable
    return coco_mask.decode(rle_obj)


def _normalize_attn_rles(attn_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ensure attention masks are a list of RLE dicts with 'counts' (str) and 'size' [H, W].
    Handles cases where counts might be stored as a string without size.
    """
    masks = attn_json.get("top_prediction_masks", [])
    shape = attn_json.get("top_prediction_masks_shape")  # [T, H, W]
    H = W = None
    if isinstance(shape, list) and len(shape) == 3:
        _, H, W = shape
    norm: List[Dict[str, Any]] = []
    for m in masks:
        if isinstance(m, dict) and "counts" in m and "size" in m:
            # Already valid
            # Ensure counts is str
            if isinstance(m["counts"], bytes):
                m = {"counts": m["counts"].decode("utf-8"), "size": m["size"]}
            norm.append(m)
        elif isinstance(m, str) and H is not None and W is not None:
            norm.append({"counts": m, "size": [H, W]})
        else:
            # Unrecognized entry; skip
            continue
    return norm


def compare_masks(attn_rles: List[Dict[str, Any]], res_rles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare masks frame-by-frame. Returns:
    - total_pixels: total number of pixels across frames
    - total_diff_pixels: sum of differing pixels across frames
    - percent_diff: total_diff_pixels / total_pixels * 100
    - min_frame_diff, max_frame_diff: min/max differing pixels in any single frame
    """
    if len(attn_rles) == 0 or len(res_rles) == 0:
        return {
            "total_pixels": 0,
            "total_diff_pixels": 0,
            "percent_diff": 0.0,
            "min_frame_diff": 0,
            "max_frame_diff": 0,
        }

    n = min(len(attn_rles), len(res_rles))
    total_pixels = 0
    total_diff = 0
    min_diff = None
    max_diff = 0

    for i in range(n):
        a = decode_rle(attn_rles[i])
        b = decode_rle(res_rles[i])
        if a is None or b is None:
            # Cannot compute without pycocotools
            return {
                "total_pixels": 0,
                "total_diff_pixels": 0,
                "percent_diff": None,
                "min_frame_diff": None,
                "max_frame_diff": None,
                "note": "pycocotools not available; mask diff not computed",
            }
        # Ensure HxW
        if a.ndim == 3:
            a = a.squeeze()
        if b.ndim == 3:
            b = b.squeeze()
        diff = np.sum(a != b)
        pix = a.size
        total_pixels += pix
        total_diff += diff
        max_diff = max(max_diff, int(diff))
        if min_diff is None:
            min_diff = int(diff)
        else:
            min_diff = min(min_diff, int(diff))

    percent = (float(total_diff) / float(total_pixels) * 100.0) if total_pixels > 0 else 0.0
    return {
        "total_pixels": int(total_pixels),
        "total_diff_pixels": int(total_diff),
        "percent_diff": percent,
        "min_frame_diff": int(min_diff) if min_diff is not None else 0,
        "max_frame_diff": int(max_diff),
    }


def get_top_traditional(res_list: List[Dict[str, Any]], video_id: int) -> Dict[str, Any]:
    vids = [r for r in res_list if r.get("video_id") == video_id]
    vids.sort(key=lambda x: x.get("score", 0), reverse=True)
    return vids[0] if vids else {}


def main():
    # Load val.json for list of videos
    val = load_json(VAL_JSON)
    videos = val.get("videos", [])

    # Load results.json once
    results = load_json(RESULTS_JSON)

    summary: Dict[str, Any] = {
        "model_dir": MODEL_DIR,
        "results_json": RESULTS_JSON,
        "attention_output_dir": ATTN_OUT_DIR,
        "videos": [],
    }

    for v in videos:
        vid = v.get("id")
        if vid is None:
            continue

        try:
            top_pred_path = ensure_attention_outputs(vid)
            top_attn = load_json(top_pred_path)
            attn_info = top_attn.get("top_prediction_info", {})
            attn_score = attn_info.get("confidence_score")
            attn_cat = attn_info.get("category_id")

            # Traditional top
            trad_top = get_top_traditional(results, vid)
            trad_score = trad_top.get("score")
            trad_cat = trad_top.get("category_id")

            # Normalize attention mask list to RLE dicts
            attn_rles = _normalize_attn_rles(top_attn)
            res_rles = trad_top.get("segmentations", [])

            mask_stats = compare_masks(attn_rles, res_rles) if attn_rles and res_rles else {
                "total_pixels": 0,
                "total_diff_pixels": 0,
                "percent_diff": 0.0,
                "min_frame_diff": 0,
                "max_frame_diff": 0,
            }

            summary["videos"].append({
                "video_id": vid,
                "attention": {
                    "category_id": attn_cat,
                    "score": attn_score,
                },
                "traditional": {
                    "category_id": trad_cat,
                    "score": trad_score,
                },
                "mask_comparison": mask_stats,
            })
        except subprocess.CalledProcessError as e:
            summary["videos"].append({
                "video_id": vid,
                "error": f"attention runner failed: {e}"
            })
        except Exception as e:
            summary["videos"].append({
                "video_id": vid,
                "error": str(e),
            })

    out_path = os.path.join(ATTN_OUT_DIR, "attention_vs_results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()


