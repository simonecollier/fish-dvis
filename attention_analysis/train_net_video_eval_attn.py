# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
except:
    pass

import logging
import os
import sys

from collections import OrderedDict
from typing import Dict

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    hooks
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# Models
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISEvaluator,
    VPSEvaluator,
    VSSEvaluator,
    add_minvis_config,
    add_dvis_config,
    add_ctvis_config,
    build_detection_test_loader,
    UniYTVISEvaluator,
    SOTDatasetMapper,
)

from dvis_daq.config import add_daq_config

#from data_scripts.ytvis_loader import register_all_ytvis_fishway
from dvis_Plus.data_video.datasets.ytvis import register_ytvis_instances
from temporal_ytvis_eval import TemporalYTVISEvaluator
from attention_extractor import AttentionExtractor

ATTENTION_EXTRACTOR = None

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator = TemporalYTVISEvaluator(dataset_name, cfg, True, output_folder)
        # Attach the attention extractor if available
        global ATTENTION_EXTRACTOR
        if ATTENTION_EXTRACTOR is not None and hasattr(evaluator, 'set_attention_extractor'):
            evaluator.set_attention_extractor(ATTENTION_EXTRACTOR)
        return evaluator

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, dataset_type):
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'vos': SOTDatasetMapper,
        }
        if dataset_type not in mapper_dict.keys():
            raise NotImplementedError
        mapper = mapper_dict[dataset_type](cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            dataset_type = cfg.DATASETS.DATASET_TYPE_TEST[idx]
            data_loader = cls.build_test_loader(cfg, dataset_name, dataset_type)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_dvis_config(cfg)
    add_ctvis_config(cfg)
    add_daq_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Register custom datasets based on config file
    # Read the config file to determine which datasets are needed
    import yaml
    with open(args.config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    train_datasets = config_data.get('DATASETS', {}).get('TRAIN', [])
    test_datasets = config_data.get('DATASETS', {}).get('TEST', [])
    
    # Determine datatype from dataset names
    datatype = None
    if train_datasets:
        if 'camera' in train_datasets[0]:
            datatype = 'camera'
        elif 'silhouette' in train_datasets[0]:
            datatype = 'silhouette'
    
    print(f"Detected datatype: {datatype}")
    
    # Register datasets based on detected datatype
    if datatype == 'camera':
        print("Registering camera datasets...")
        register_ytvis_instances(
            "ytvis_fishway_val_camera_fold4",
            {},
            #"/data/fishway_ytvis/val_vid_66.json",
            "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/fold4/checkpoint_0006059/val_fold4_all_frames.json",
            "/data/fishway_ytvis/all_videos"
        )
    elif datatype == 'silhouette':
        print("Registering silhouette datasets...")
        register_ytvis_instances(
            "ytvis_fishway_val_silhouette_fold2",
            {},
            #"/data/fishway_ytvis/val_vid_66.json",
            "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/fold2/checkpoint_0005655/val_fold2_all_frames.json",
            "/data/fishway_ytvis/all_videos_mask"
        )
    else:
        print("Warning: Could not determine datatype from config file. Registering all datasets.")
        # Fallback: register all datasets if datatype cannot be determined
        register_ytvis_instances(
            "ytvis_fishway_val_camera_fold4",
            {},
            #"/data/fishway_ytvis/val_vid_66.json",
            "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/fold4/checkpoint_0006059/val_fold4_all_frames.json",
            "/data/fishway_ytvis/all_videos"
        )
        register_ytvis_instances(
            "ytvis_fishway_val_silhouette_fold2",
            {},
            #"/data/fishway_ytvis/val_vid_66.json",
            "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/fold2/checkpoint_0005655/val_fold2_all_frames.json",
            "/data/fishway_ytvis/all_videos_mask"
        )

    # register_ytvis_instances(
    #     "ytvis_fishway_val",
    #     {},
    #     "/data/fishway_ytvis/val.json",
    #     "/data/fishway_ytvis/all_videos"
    # )

    # Add debug flag to config
    cfg.DEBUG = args.debug
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="minvis")
    
    # Debug prints for window inference settings (only if debug is enabled)
    if cfg.DEBUG:
        print(f"[DEBUG] WINDOW_INFERENCE: {cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE}")
        print(f"[DEBUG] WINDOW_SIZE: {cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE}")
        print(f"[DEBUG] MAX_NUM: {cfg.MODEL.MASK_FORMER.TEST.MAX_NUM}")
        print(f"[DEBUG] NUM_OBJECT_QUERIES: {cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES}")
    
    return cfg


def main(args):
    # Memory optimization settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # Initialize attention extractor after weights are loaded
        try:
            global ATTENTION_EXTRACTOR
            from temporal_ytvis_eval import TemporalYTVISEvaluator  # ensure import for hooks
            # Create the extractor using the model already built and loaded
            # Enable save_immediately_from_hook to prevent RAM accumulation
            ATTENTION_EXTRACTOR = AttentionExtractor(
                model, 
                cfg.OUTPUT_DIR,
                extract_backbone=False,  # Set to False to skip backbone attention extraction (saves memory)
                extract_refiner=True,    # Set to False to skip refiner attention extraction
                save_immediately_from_hook=True,  # Save directly from hook, no accumulation
                window_size=cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE  # Pass window_size from config
            )
        except Exception as e:
            print(f"[WARN] Attention extractor could not be initialized: {e}")
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="Enable debug prints during training/evaluation")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
