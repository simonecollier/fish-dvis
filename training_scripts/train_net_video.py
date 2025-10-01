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
except:
    pass

import copy
import itertools
import logging
import os
import sys

# Add the project root to Python path to import our custom modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import gc

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
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
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
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
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    UniYTVISEvaluator,
    SOTDatasetMapper,
)

from dvis_daq.config import add_daq_config

#from data_scripts.ytvis_loader import register_all_ytvis_fishway
from data_scripts.ytvis_loader import register_ytvis_instances

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

        if cfg.MODEL.MASK_FORMER.TEST.TASK == "vos":
            return None

        evaluator_dict = {'vis': YTVISEvaluator, 'vss': VSSEvaluator, 'vps': VPSEvaluator, 'mots': UniYTVISEvaluator}
        assert cfg.MODEL.MASK_FORMER.TEST.TASK in evaluator_dict.keys()
        return evaluator_dict[cfg.MODEL.MASK_FORMER.TEST.TASK](dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        assert len(cfg.DATASETS.DATASET_RATIO) == len(cfg.DATASETS.TRAIN) ==\
               len(cfg.DATASETS.DATASET_NEED_MAP) == len(cfg.DATASETS.DATASET_TYPE)
        mappers = []
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'image_instance': CocoClipDatasetMapper,
        }
        for d_i, (dataset_name, dataset_type, dataset_need_map) in \
                enumerate(zip(cfg.DATASETS.TRAIN, cfg.DATASETS.DATASET_TYPE, cfg.DATASETS.DATASET_NEED_MAP)):
            if dataset_type not in mapper_dict.keys():
                raise NotImplementedError
            _mapper = mapper_dict[dataset_type]
            mappers.append(
                _mapper(cfg, is_train=True, is_tgt=not dataset_need_map, src_dataset_name=dataset_name, )
            )
        assert len(mappers) > 0, "No dataset is chosen!"

        if len(mappers) == 1:
            mapper = mappers[0]
            return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
        else:
            loaders = [
                build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN)
            ]
            combined_data_loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)
            return combined_data_loader

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
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        skip_params = cfg.MODEL.VIDEO_HEAD.SKIP_PARAMS

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

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

    # Get test frame stride from config
    test_frame_stride = getattr(cfg.INPUT, 'TEST_SAMPLING_FRAME_STRIDE', 
                               getattr(cfg.INPUT, 'SAMPLING_FRAME_STRIDE', 1))
    
    # Ensure stride-adjusted validation JSON exists if needed
    data_scripts_path = os.path.join(project_root, 'data_scripts')
    if data_scripts_path not in sys.path:
        sys.path.insert(0, data_scripts_path)
    
    try:
        from create_stride_adjusted_val_json import ensure_stride_adjusted_json
        
        val_json_path = ensure_stride_adjusted_json(
            original_json_path="/data/fishway_ytvis/val.json",
            frame_stride=test_frame_stride,
            model_output_dir=cfg.OUTPUT_DIR,
            verbose=True
        )
        
        # CRITICAL: If using stride-adjusted JSON, disable TEST frame stride in the data loader
        # to prevent double-striding (JSON is already stride-adjusted)
        # NOTE: We only modify TEST_SAMPLING_FRAME_STRIDE to preserve training stride behavior
        if test_frame_stride > 1 and "stride" in val_json_path:
            print(f"✓ Using stride-adjusted JSON - disabling TEST frame stride to prevent double-striding")
            print(f"  → Training stride ({cfg.INPUT.SAMPLING_FRAME_STRIDE}) remains unchanged")
            print(f"  → Test stride changed: {test_frame_stride} → 1")
            cfg.INPUT.TEST_SAMPLING_FRAME_STRIDE = 1
        
    except ImportError as e:
        print(f"Warning: Could not import stride adjustment utility: {e}")
        print("Using original validation JSON")
        val_json_path = "/data/fishway_ytvis/val.json"
    
    # Register multiple datasets - you can reference any of these in your config
    datasets_to_register = {
        # Fishway datasets
        "ytvis_fishway_train": {
            "json_path": "/data/fishway_ytvis/train.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val": {
            "json_path": "/data/fishway_ytvis/val.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_stride1": {
            "json_path": "/data/fishway_ytvis/train_stride1.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride1": {
            "json_path": "/data/fishway_ytvis/val_stride1.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_camera0": {
            "json_path": "/data/fishway_ytvis/train_camera0.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_camera0": {
            "json_path": "/data/fishway_ytvis/val_camera0.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride1_all": {
            "json_path": "/data/fishway_ytvis/val_stride1_all.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_stride2": {
            "json_path": "/data/fishway_ytvis/train_stride2.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride2": {
            "json_path": "/data/fishway_ytvis/val_stride2.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_stride3": {
            "json_path": "/data/fishway_ytvis/train_stride3.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride3": {
            "json_path": "/data/fishway_ytvis/val_stride3.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_stride4": {
            "json_path": "/data/fishway_ytvis/train_stride4.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride4": {
            "json_path": "/data/fishway_ytvis/val_stride4.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_stride5": {
            "json_path": "/data/fishway_ytvis/train_stride5.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride5": {
            "json_path": "/data/fishway_ytvis/val_stride5.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_train_stride6": {
            "json_path": "/data/fishway_ytvis/train_stride6.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        "ytvis_fishway_val_stride6": {
            "json_path": "/data/fishway_ytvis/val_stride6.json",
            "image_root": "/data/fishway_ytvis/all_videos"
        },
        # Add more datasets here as needed
        # "ytvis_custom_dataset": {
        #     "json_path": "/path/to/custom/dataset.json",
        #     "image_root": "/path/to/custom/images"
        # },
    }
    
    # Check if config specifies which datasets to register (optional)
    datasets_to_use = getattr(cfg, 'DATASETS_TO_REGISTER', None)
    if datasets_to_use is not None:
        # Only register datasets specified in config
        datasets_to_register = {k: v for k, v in datasets_to_register.items() if k in datasets_to_use}
        print(f"Registering only specified datasets: {datasets_to_use}")

    # During evaluation, only register the test datasets referenced by the config
    if args.eval_only:
        test_datasets = list(getattr(cfg.DATASETS, 'TEST', []))
        if test_datasets:
            datasets_to_register = {k: v for k, v in datasets_to_register.items() if k in test_datasets}
            print(f"Eval-only run: registering only test datasets: {test_datasets}")

    # Apply VAL_JSON_OVERRIDE before any registration, so we don't touch missing /data paths
    val_json_override = os.environ.get("VAL_JSON_OVERRIDE")
    if val_json_override and datasets_to_register:
        print(f"VAL_JSON_OVERRIDE detected -> {val_json_override}")
        for name in list(datasets_to_register.keys()):
            if not args.eval_only or name in getattr(cfg.DATASETS, 'TEST', []):
                datasets_to_register[name] = {
                    **datasets_to_register[name],
                    "json_path": val_json_override,
                }
        # Also set TEST_SAMPLING_FRAME_STRIDE to 1 when overriding (to avoid double-stride if any)
        if hasattr(cfg.INPUT, 'TEST_SAMPLING_FRAME_STRIDE'):
            cfg.INPUT.TEST_SAMPLING_FRAME_STRIDE = 1
    
    # Register all datasets (skip if JSON does not exist)
    for dataset_name, dataset_info in datasets_to_register.items():
        json_path = dataset_info["json_path"]
        image_root = dataset_info["image_root"]
        if not os.path.exists(json_path):
            print(f"[WARN] Skipping dataset '{dataset_name}' - JSON not found: {json_path}")
            continue
        register_ytvis_instances(
            dataset_name,
            {},
            json_path,
            image_root,
        )
        print(f"Registered dataset: {dataset_name}")

    # (Override already applied prior to registration if present)

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
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Add memory cleanup hook to prevent OOM
    class MemoryCleanupHook(hooks.HookBase):
        def __init__(self, period=50):
            super().__init__()
            self.period = period
            self.iter = 0
            
        def after_step(self):
            self.iter += 1
            if self.iter % self.period == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    trainer.register_hooks([MemoryCleanupHook(period=50)])

    # # Add periodic validation evaluation
    # Set to None to disable evaluation during training (to avoid OOM)
    # eval_period = None  # Set to 50 to enable evaluation, None to disable
    # def eval_and_clear():
    #     # Clear memory before evaluation
    #     torch.cuda.empty_cache()
    #     gc.collect()
        
    #     try:
    #         results = Trainer.test(cfg, trainer.model)
    #         # Clear memory after evaluation
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #         return results
    #     except torch.cuda.OutOfMemoryError:
    #         print("OOM during evaluation - skipping this evaluation")
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #         return {}

    # if eval_period is not None:
    #     trainer.register_hooks([
    #         hooks.EvalHook(eval_period, eval_and_clear)
    #     ])

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="Enable debug prints during training/evaluation")
    args = parser.parse_args()
    #args.dist_url = 'tcp://127.0.0.1:50263'
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
