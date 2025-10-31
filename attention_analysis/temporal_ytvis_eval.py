import contextlib
import io
import copy
import os
import sys
import json
import logging
from collections import OrderedDict

import numpy as np
# NumPy compatibility shim for deprecated aliases used in YTVIS eval
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
import pycocotools.mask as mask_util
import torch

import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

# Ensure project root and DVIS_Plus are importable when running from attention_analysis
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DVIS_PLUS_ROOT = os.path.join(_PROJECT_ROOT, "DVIS_Plus")
for p in (_PROJECT_ROOT, _DVIS_PLUS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the project evaluator and helpers
from mask2former_video.data_video.ytvis_eval import YTVISEvaluator


def _instances_to_coco_json_video_with_refiner(inputs, outputs):
    """
    Convert model outputs to YTVIS-style JSON entries, adding refiner_id.
    Expects outputs to contain: pred_scores, pred_labels, pred_masks, pred_ids (aligned).
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]
    refiner_ids = outputs.get("pred_ids", None)

    ytvis_results = []
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, masks)):
        segms = [
            mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in m
        ]
        for rle in segms:
            rle["counts"] = rle["counts"].decode("utf-8")

        # Build with desired key order; place refiner_id before segmentations
        res = OrderedDict()
        res["video_id"] = video_id
        res["score"] = s
        res["category_id"] = l
        if refiner_ids is not None and instance_id < len(refiner_ids):
            try:
                res["refiner_id"] = int(refiner_ids[instance_id])
            except Exception:
                pass
        res["segmentations"] = segms
        ytvis_results.append(res)

    return ytvis_results


class TemporalYTVISEvaluator(YTVISEvaluator):
    """
    YTVIS evaluator that writes a sidecar results_temporal.json containing
    a replica of results.json augmented with refiner_id per prediction.
    """

    def reset(self):
        super().reset()
        self._predictions_temporal = []

    def process(self, inputs, outputs):
        # Standard predictions for metrics and results.json
        super().process(inputs, outputs)
        # Augmented predictions for results_temporal.json
        try:
            aug = _instances_to_coco_json_video_with_refiner(inputs, outputs)
            self._predictions_temporal.extend(aug)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"TemporalYTVISEvaluator process() failed to add refiner ids: {e}")

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(sum(predictions, []))

            temporal_predictions = comm.gather(self._predictions_temporal, dst=0)
            temporal_predictions = list(sum(temporal_predictions, []))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            temporal_predictions = self._predictions_temporal

        if len(predictions) == 0:
            logging.getLogger(__name__).warning("[TemporalYTVISEvaluator] Did not receive valid predictions.")
            return {}

        # Let the base class handle standard saving and metrics
        self._results = OrderedDict()
        self._eval_predictions(predictions)

        # Apply the same category ID unmapping to temporal predictions
        # (convert from contiguous IDs 0-4 to dataset IDs 1-5)
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            if min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1:
                reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
                for result in temporal_predictions:
                    category_id = result["category_id"]
                    if category_id < num_classes:
                        result["category_id"] = reverse_id_mapping[category_id]

        # Write the augmented replica
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            temporal_path = os.path.join(self._output_dir, "results_temporal.json")
            with PathManager.open(temporal_path, "w") as f:
                f.write(json.dumps(temporal_predictions))
                f.flush()

            # Also write a compact summary of all refiner ids (overall and per video)
            try:
                all_ids = []
                per_video = {}
                for item in temporal_predictions:
                    vid = item.get("video_id")
                    rid = item.get("refiner_id")
                    if rid is None:
                        continue
                    all_ids.append(rid)
                    if vid not in per_video:
                        per_video[vid] = []
                    per_video[vid].append(rid)

                all_ids_unique = sorted({int(x) for x in all_ids}) if len(all_ids) else []
                per_video_unique = {
                    str(k): sorted({int(x) for x in v}) for k, v in per_video.items()
                }
                summary = {
                    "refiner_ids_all": all_ids_unique,
                    "refiner_ids_per_video": per_video_unique,
                }
                temporal_summary_path = os.path.join(self._output_dir, "results_temporal_summary.json")
                with PathManager.open(temporal_summary_path, "w") as fsum:
                    fsum.write(json.dumps(summary))
                    fsum.flush()
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to write results_temporal_summary.json: {e}"
                )

        # Return the same metrics dict as base
        return copy.deepcopy(self._results)


