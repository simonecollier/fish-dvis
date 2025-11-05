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

import copy
import itertools
import logging
import os
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
from dvis_Plus.data_video.datasets.ytvis import register_ytvis_instances

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

    # Register custom datasets based on config file
    # Read the config file to determine which datasets are needed
    import yaml
    import re
    with open(args.config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    train_datasets = config_data.get('DATASETS', {}).get('TRAIN', [])
    test_datasets = config_data.get('DATASETS', {}).get('TEST', [])
    
    # Helper function to extract stride and base info from dataset name
    def parse_dataset_name(dataset_name):
        """Extract stride value and base dataset info from dataset name.
        
        Returns:
            tuple: (base_name, is_train, datatype, stride_value, stride_suffix)
            - base_name: base name without stride suffix (e.g., 'ytvis_fishway_train_silhouette')
            - is_train: True if train, False if val/test
            - datatype: 'camera' or 'silhouette'
            - stride_value: stride value (None if no stride)
            - stride_suffix: '_stride{N}' or ''
        """
        # Match stride pattern: _stride{N} at the end
        stride_match = re.search(r'_stride(\d+)$', dataset_name)
        stride_value = int(stride_match.group(1)) if stride_match else None
        stride_suffix = stride_match.group(0) if stride_match else ''
        base_name = dataset_name[:-len(stride_suffix)] if stride_suffix else dataset_name
        
        # Determine if train or val
        is_train = 'train' in base_name
        is_val = 'val' in base_name
        
        # Determine datatype
        datatype = 'silhouette' if 'silhouette' in base_name else 'camera'
        
        return base_name, is_train, datatype, stride_value, stride_suffix
    
    # Collect all unique datasets that need to be registered
    all_datasets = set(train_datasets + test_datasets)
    registered_datasets = set()
    
    # Register each dataset
    for dataset_name in all_datasets:
        if dataset_name in registered_datasets:
            continue
            
        base_name, is_train, datatype, stride_value, stride_suffix = parse_dataset_name(dataset_name)
        
        # Determine JSON path
        if is_train:
            json_name = 'train'
            image_root = '/data/fishway_ytvis/all_videos' if datatype == 'camera' else '/data/fishway_ytvis/all_videos_mask'
        else:
            json_name = 'val'
            image_root = '/data/fishway_ytvis/all_videos' if datatype == 'camera' else '/data/fishway_ytvis/all_videos_mask'
        
        # Use strided JSON if stride is specified
        if stride_value:
            json_path = f"/data/fishway_ytvis/{json_name}_stride{stride_value}.json"
            print(f"Registering {dataset_name} with strided JSON: {json_path}")
        else:
            json_path = f"/data/fishway_ytvis/{json_name}.json"
            print(f"Registering {dataset_name} with JSON: {json_path}")
        
        register_ytvis_instances(dataset_name, {}, json_path, image_root)
        registered_datasets.add(dataset_name)

    # register_ytvis_instances(
    #     "ytvis_fishway_train",
    #     {},
    #     "/data/fishway_ytvis/train.json",
    #     "/data/fishway_ytvis/all_videos"
    # )
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
