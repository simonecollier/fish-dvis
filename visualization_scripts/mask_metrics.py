import json
import numpy as np
from pycocotools import mask as maskUtils
from collections import defaultdict
import csv
from skimage import measure
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def decode_rle(rle):
    # rle: dict with 'size' and 'counts'
    return maskUtils.decode(rle)

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 1.0

def boundary_f_measure(pred_mask, gt_mask, bound_th=2):
    # Compute boundary F-measure for binary masks
    pred_contour = measure.find_contours(pred_mask, 0.5)
    gt_contour = measure.find_contours(gt_mask, 0.5)
    if not pred_contour or not gt_contour:
        return 0.0
    pred_points = np.concatenate(pred_contour)
    gt_points = np.concatenate(gt_contour)
    # Compute distances
    from scipy.spatial.distance import cdist
    dists_pred_to_gt = cdist(pred_points, gt_points)
    dists_gt_to_pred = cdist(gt_points, pred_points)
    # Precision: fraction of pred points within bound_th of any gt point
    precision = np.mean(np.min(dists_pred_to_gt, axis=1) <= bound_th)
    # Recall: fraction of gt points within bound_th of any pred point
    recall = np.mean(np.min(dists_gt_to_pred, axis=1) <= bound_th)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_ap_at_iou(pred_by_video, gt_dict, iou_threshold, category_id):
    """
    Compute Average Precision (AP) for a single category at a given IoU threshold.
    """
    all_scores = []
    all_matches = []
    num_gt_total = 0

    video_ids = set(list(gt_dict.keys()) + list(pred_by_video.keys()))

    for video_id in video_ids:
        gt_tracks = [ann for ann in gt_dict.get(video_id, []) if ann['category_id'] == category_id]
        pred_tracks = [pred for pred in pred_by_video.get(video_id, []) if pred['category_id'] == category_id]

        num_gt_total += len(gt_tracks)

        if not pred_tracks:
            continue

        gt_matched = {i: False for i in range(len(gt_tracks))}
        
        # Sort predictions by score
        pred_tracks.sort(key=lambda p: p.get('score', 0.0), reverse=True)
        
        if not gt_tracks:
            all_scores.extend([p.get('score', 0.0) for p in pred_tracks])
            all_matches.extend([0] * len(pred_tracks)) # All are FPs
            continue

        iou_matrix = np.zeros((len(gt_tracks), len(pred_tracks)))
        for i, gt_ann in enumerate(gt_tracks):
            gt_segs = gt_ann['segmentations']
            for j, pred_ann in enumerate(pred_tracks):
                pred_segs = pred_ann['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                        continue
                    if pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                        continue
                    
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                iou_matrix[i, j] = mean_iou
        
        for j in range(len(pred_tracks)):
            all_scores.append(pred_tracks[j].get('score', 0.0))
            
            best_iou = -1
            best_gt_idx = -1
            for i in range(len(gt_tracks)):
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = i
            
            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    all_matches.append(1)  # TP
                else:
                    all_matches.append(0)  # FP (duplicate detection)
            else:
                all_matches.append(0)  # FP
    
    if num_gt_total == 0:
        return 1.0 if not all_scores else 0.0

    if not all_scores:
        return 0.0

    sorted_indices = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[sorted_indices]
    
    tp = np.cumsum(all_matches)
    fp = np.cumsum(all_matches == 0)

    recalls = tp / num_gt_total
    precisions = tp / (tp + fp)

    # Compute AP using 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p / 101
        
    return ap

def main(results_json, val_json, csv_path="mask_metrics.csv", cm_plot_path="confusion_matrix.png"):
    preds = load_json(results_json)
    gts = load_json(val_json)

    gt_category_ids = {cat['id'] for cat in gts['categories']}
    gt_category_names = {cat['id']: cat['name'] for cat in gts['categories']}

    # Build GT dict: {video_id: [annotations]}
    gt_dict = defaultdict(list)
    gt_meta = {}
    for ann in gts['annotations']:
        gt_dict[ann['video_id']].append(ann)
        gt_meta[(ann['video_id'], ann['id'])] = (ann['category_id'], ann.get('file_name', ''))

    # Group predictions by video_id
    pred_by_video = defaultdict(list)
    for pred in preds:
        if pred.get('category_id') in gt_category_ids:
            vid = pred['video_id']
            pred_by_video[vid].append(pred)

    all_frame_ious = []
    all_frame_fmeasures = []
    csv_rows = []
    video_ious = {}
    video_fmeasures = {}

    video_ids = set(list(gt_dict.keys()) + list(pred_by_video.keys()))
    # Prepare per-video prediction info
    video_pred_info = {}
    for video_id in video_ids:
        preds_for_video = pred_by_video.get(video_id, [])
        if preds_for_video:
            best_pred = max(preds_for_video, key=lambda p: p.get('score', 0.0))
            pred_cat_id = best_pred.get('category_id', None)
            pred_cat_name = gt_category_names.get(pred_cat_id, f"ID {pred_cat_id}")
            pred_score = best_pred.get('score', 0.0)
        else:
            pred_cat_id = None
            pred_cat_name = None
            pred_score = None
        gt_cat_ids = sorted(set(ann['category_id'] for ann in gt_dict.get(video_id, [])))
        gt_cat_names = [gt_category_names.get(cid, f"ID {cid}") for cid in gt_cat_ids]
        video_pred_info[video_id] = {
            'predicted_category_id': pred_cat_id,
            'predicted_category_name': pred_cat_name,
            'predicted_score': pred_score,
            'gt_category_ids': gt_cat_ids,
            'gt_category_names': gt_cat_names
        }

    for video_id in tqdm(video_ids, desc="Processing videos"):
        video_frame_ious_per_video = []
        video_frame_fmeasures_per_video = []

        for cat_id in gt_category_ids:
            gt_tracks = [ann for ann in gt_dict.get(video_id, []) if ann['category_id'] == cat_id]
            pred_tracks = [pred for pred in pred_by_video.get(video_id, []) if pred['category_id'] == cat_id]

            if not gt_tracks or not pred_tracks:
                continue
            # Build cost matrix: rows=GT, cols=Pred, value=1-mean IoU
            num_gt = len(gt_tracks)
            num_pred = len(pred_tracks)
            cost_matrix = np.ones((num_gt, num_pred))
            iou_matrix = np.zeros((num_gt, num_pred))
            f_matrix = np.zeros((num_gt, num_pred))
            for i, gt_ann in enumerate(gt_tracks):
                gt_segs = gt_ann['segmentations']
                # gt_height = gt_ann['height']
                # gt_width = gt_ann['width']
                for j, pred_ann in enumerate(pred_tracks):
                    pred_segs = pred_ann['segmentations']
                    # pred_height = pred_ann['height']
                    # pred_width = pred_ann['width']
                    length = min(len(pred_segs), len(gt_segs))
                    frame_ious = []
                    frame_fmeasures = []
                    for frame_idx in range(length):
                        pred_rle = pred_segs[frame_idx]
                        gt_rle = gt_segs[frame_idx]
                        # Always use fixed shape for all-zeros mask
                        mask_shape = (960, 1280)
                        if pred_rle is None and gt_rle is not None:
                            pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                            gt_mask = decode_rle(gt_rle)
                        elif gt_rle is None and pred_rle is not None:
                            gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                            pred_mask = decode_rle(pred_rle)
                        elif pred_rle is None and gt_rle is None:
                            pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                            gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                        else:
                            pred_mask = decode_rle(pred_rle)
                            gt_mask = decode_rle(gt_rle)
                        iou = compute_iou(pred_mask, gt_mask)
                        fmeasure = boundary_f_measure(pred_mask, gt_mask)
                        frame_ious.append(iou)
                        frame_fmeasures.append(fmeasure)
                    mean_iou = np.mean(frame_ious) if frame_ious else 0.0
                    mean_f = np.mean(frame_fmeasures) if frame_fmeasures else 0.0
                    cost_matrix[i, j] = 1 - mean_iou
                    iou_matrix[i, j] = mean_iou
                    f_matrix[i, j] = mean_f
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
            # For each matched pair, compute per-frame and per-track metrics
            
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                gt_ann = gt_tracks[gt_idx]
                pred_ann = pred_tracks[pred_idx]
                gt_segs = gt_ann['segmentations']
                pred_segs = pred_ann['segmentations']
                # gt_height = gt_ann['height']
                # gt_width = gt_ann['width']
                # pred_height = pred_ann['height']
                # pred_width = pred_ann['width']
                length = min(len(pred_segs), len(gt_segs))
                frame_ious = []
                frame_fmeasures = []
                category_id, file_name = gt_meta[(video_id, gt_ann['id'])]
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    # Always use fixed shape for all-zeros mask
                    mask_shape = (960, 1280)
                    if pred_rle is None and gt_rle is not None:
                        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                        gt_mask = decode_rle(gt_rle)
                    elif gt_rle is None and pred_rle is not None:
                        gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                        pred_mask = decode_rle(pred_rle)
                    elif pred_rle is None and gt_rle is None:
                        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                        gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                    else:
                        pred_mask = decode_rle(pred_rle)
                        gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    fmeasure = boundary_f_measure(pred_mask, gt_mask)
                    frame_ious.append(iou)
                    frame_fmeasures.append(fmeasure)
                    all_frame_ious.append(iou)
                    all_frame_fmeasures.append(fmeasure)
                    csv_rows.append({
                        "video_id": video_id,
                        "category_id": category_id,
                        "file_name": file_name,
                        "frame_IoU": iou,
                        "frame_boundary_Fmeasure": fmeasure,
                        "video_IoU": None,  # placeholder
                        "video_boundary_Fmeasure": None,  # placeholder
                        "dataset_IoU": None,  # placeholder
                        "dataset_boundary_Fmeasure": None  # placeholder
                    })
                mean_iou = np.mean(frame_ious) if frame_ious else 0.0
                mean_f = np.mean(frame_fmeasures) if frame_fmeasures else 0.0
                video_frame_ious_per_video.append(mean_iou)
                video_frame_fmeasures_per_video.append(mean_f)

        video_ious[video_id] = np.mean(video_frame_ious_per_video) if video_frame_ious_per_video else 0.0
        video_fmeasures[video_id] = np.mean(video_frame_fmeasures_per_video) if video_frame_fmeasures_per_video else 0.0

    dataset_iou = np.mean(list(video_ious.values())) if video_ious else 0.0
    dataset_fmeasure = np.mean(list(video_fmeasures.values())) if video_fmeasures else 0.0

    # Compute mAP@0.5, mAP@0.75, mAP@0.25, and mAP@0.95
    map25_scores = []
    map50_scores = []
    map75_scores = []
    map95_scores = []
    ap_per_category = {}
    for cat_id in gt_category_ids:
        ap25 = compute_ap_at_iou(pred_by_video, gt_dict, 0.25, cat_id)
        ap50 = compute_ap_at_iou(pred_by_video, gt_dict, 0.5, cat_id)
        ap75 = compute_ap_at_iou(pred_by_video, gt_dict, 0.75, cat_id)
        ap95 = compute_ap_at_iou(pred_by_video, gt_dict, 0.95, cat_id)
        map25_scores.append(ap25)
        map50_scores.append(ap50)
        map75_scores.append(ap75)
        map95_scores.append(ap95)
        ap_per_category[cat_id] = {
            'AP@0.25': ap25,
            'AP@0.5': ap50,
            'AP@0.75': ap75,
            'AP@0.95': ap95,
        }
        cat_name = gt_category_names.get(cat_id, f"ID {cat_id}")
        print(f"Category '{cat_name}': AP@0.25 = {ap25:.4f}, AP@0.5 = {ap50:.4f}, AP@0.75 = {ap75:.4f}, AP@0.95 = {ap95:.4f}")

    map25 = np.mean(map25_scores) if map25_scores else 0.0
    map50 = np.mean(map50_scores) if map50_scores else 0.0
    map75 = np.mean(map75_scores) if map75_scores else 0.0
    map95 = np.mean(map95_scores) if map95_scores else 0.0

    # Fill in video and dataset metrics in csv_rows
    idx = 0
    for row in csv_rows:
        video_id = row["video_id"]
        row["video_IoU"] = video_ious[video_id]
        row["video_boundary_Fmeasure"] = video_fmeasures[video_id]
        row["dataset_IoU"] = dataset_iou
        row["dataset_boundary_Fmeasure"] = dataset_fmeasure
        row["mAP@0.25"] = map25
        row["mAP@0.5"] = map50
        row["mAP@0.75"] = map75
        row["mAP@0.95"] = map95
        # Add per-category APs
        cat_id = row["category_id"]
        ap_cat = ap_per_category.get(cat_id, {'AP@0.25': None, 'AP@0.5': None, 'AP@0.75': None, 'AP@0.95': None})
        row["AP@0.25"] = ap_cat['AP@0.25']
        row["AP@0.5"] = ap_cat['AP@0.5']
        row["AP@0.75"] = ap_cat['AP@0.75']
        row["AP@0.95"] = ap_cat['AP@0.95']
        # Add prediction info
        pred_info = video_pred_info[video_id]
        row["predicted_category_id"] = pred_info['predicted_category_id']
        row["predicted_category_name"] = pred_info['predicted_category_name']
        row["predicted_score"] = pred_info['predicted_score']
        row["gt_category_id"] = ','.join(str(x) for x in pred_info['gt_category_ids'])
        row["gt_category_name"] = ','.join(pred_info['gt_category_names'])
        idx += 1

    # Print per-video and dataset metrics
    for video_id in video_ious:
        pred_info = video_pred_info[video_id]
        print(f"Video {video_id}: Mean IoU = {video_ious[video_id]:.4f}, Mean Boundary F = {video_fmeasures[video_id]:.4f}, "
              f"Predicted Cat: {pred_info['predicted_category_id']} ({pred_info['predicted_category_name']}), "
              f"Score: {pred_info['predicted_score']}, "
              f"GT Cat(s): {pred_info['gt_category_ids']} ({pred_info['gt_category_names']})")
    print(f"Overall Dataset: Mean IoU = {dataset_iou:.4f}, Mean Boundary F = {dataset_fmeasure:.4f}")
    print(f"mAP@0.25: {map25:.4f}, mAP@0.5: {map50:.4f}, mAP@0.75: {map75:.4f}, mAP@0.95: {map95:.4f}")

    # Save CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "category_id", "file_name", "frame_IoU", "frame_boundary_Fmeasure", "video_IoU", "video_boundary_Fmeasure", "dataset_IoU", "dataset_boundary_Fmeasure", "mAP@0.25", "mAP@0.5", "mAP@0.75", "mAP@0.95", "AP@0.25", "AP@0.5", "AP@0.75", "AP@0.95", "predicted_category_id", "predicted_category_name", "predicted_score", "gt_category_id", "gt_category_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # --- Confusion Matrix Calculation ---
    # For each GT object, find the best-matching prediction (highest IoU, any class)
    y_true = []
    y_pred = []
    for video_id in video_ids:
        gt_anns = gt_dict.get(video_id, [])
        preds = pred_by_video.get(video_id, [])
        for gt_ann in gt_anns:
            gt_cat = gt_ann['category_id']
            gt_segs = gt_ann['segmentations']
            best_iou = -1
            best_pred_cat = None
            for pred in preds:
                pred_cat = pred['category_id']
                pred_segs = pred['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    mask_shape = (960, 1280)
                    if pred_rle is None and gt_rle is not None:
                        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                        gt_mask = decode_rle(gt_rle)
                    elif gt_rle is None and pred_rle is not None:
                        gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                        pred_mask = decode_rle(pred_rle)
                    elif pred_rle is None and gt_rle is None:
                        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                        gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                    else:
                        pred_mask = decode_rle(pred_rle)
                        gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_pred_cat = pred_cat
            y_true.append(gt_cat)
            y_pred.append(best_pred_cat if best_pred_cat is not None else -1)
    # Confusion matrix
    labels = sorted(list(gt_category_ids))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[gt_category_names[l] for l in labels])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close(fig)

def plot_confusion_matrix_from_csv(csv_path, output_path="confusion_matrix_from_csv.png"):
    """
    Plots a confusion matrix using only the first frame of each video from the mask_metrics.csv file.
    Compares predicted_category_name to gt_category_name.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Get the first frame for each video (assuming sorted by video_id and frame order)
    first_frames = df.groupby('video_id').first().reset_index()
    print(first_frames.head())
    # Extract predicted and ground truth category names
    y_true = first_frames['gt_category_name']
    y_pred = first_frames['predicted_category_name']

    # Get sorted list of all unique categories
    all_categories = sorted(set(y_true) | set(y_pred))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_categories)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_categories)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix (First Frame per Video)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_ap_per_category_from_csv(csv_path, output_path="AP_per_category.png"):
    """
    Reads mask_metrics.csv and plots the AP@0.25, AP@0.5, AP@0.75, AP@0.95 for each category in the dataset (grouped by gt_category_name).
    The x-axis is the category, the y-axis is AP, and each AP level is a different colored point.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    ap_columns = ["AP@0.25", "AP@0.5", "AP@0.75", "AP@0.95"]
    grouped = df.groupby('gt_category_name')[ap_columns].first().reset_index()

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, ap_col in enumerate(ap_columns):
        plt.scatter(grouped['gt_category_name'], grouped[ap_col], label=ap_col, marker=markers[i], color=colors[i], s=80)
    plt.xlabel('Category')
    plt.ylabel('Average Precision (AP)')
    plt.title('AP per Category at Different IoU Thresholds')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(title='IoU Threshold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main(results_json='/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/inference/results.json', 
         val_json='/data/fishway_ytvis/val_trimmed.json', 
         csv_path='/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/inference/mask_metrics.csv',
         cm_plot_path='/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/inference/confusion_matrix.png')
    
    # plot_confusion_matrix_from_csv(csv_path='/home/simone/fish-dvis/dvis-model-outputs/trained_models/dvis_daq_vitl_offline_80vids_10k/iter_8199/inference/mask_metrics.csv',
    #                                output_path='/home/simone/fish-dvis/dvis-model-outputs/trained_models/dvis_daq_vitl_offline_80vids_10k/iter_8199/inference/confusion_matrix.png')

    plot_ap_per_category_from_csv(csv_path='/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/inference/mask_metrics.csv', 
                                   output_path='/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/inference/AP_per_category.png')