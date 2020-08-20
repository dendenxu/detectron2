#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

from tqdm import tqdm
import logging
import coloredlogs
import numpy as np
import cv2
import os
import torch
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from imantics import Mask, Polygons
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
from detectron2.structures import BoxMode
from PIL import ImageFile
import json
from detectron2.structures import Boxes, BoxMode, PolygonMasks
import pycocotools.mask as mask_util
ImageFile.LOAD_TRUNCATED_IMAGES = True
coloredlogs.install("INFO")
log = logging.getLogger(__name__)


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
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
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_hair()  # this is some customized logic to register our hair datasets

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def register_hair():
    def get_hair_dicts(dir_path, dir_name="train", dataset_name="large", class_name="hair"):
        log.info(f"{type_path}, {d}, {dataset_name}, {class_name}")

        name_prefix = dataset_name + "_" + class_name + "_" + dir_name
        json_file = name_prefix + ".json"
        json_path = os.path.join(dir_path, json_file)

        log.info(f"Loading annotation json file from {json_path}")

        with open(json_path, "r") as f:
            json_dict = json.load(f)

        log.info(f"Successfully loaded json file from {json_path}")
        dataset_dicts = []
        for idx in tqdm(range(len(json_dict["images"])), "Accessing json annotation: "):
            anno = json_dict["annotations"][idx]
            img = json_dict["images"][idx]
            # anno["bbox_mode"] = BoxMode.XYXY_ABS
            # box = anno["bbox"]
            # anno["bbox"] = np.asarray([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            # anno["bbox"] = np.reshape(anno["bbox"], [2, 2])

            assert anno["image_id"] == img["id"], "Unable to match annotation image_id with image_id"
            dataset_dicts.append({
                "file_name": os.path.join(dir_path, img["file_name"]),
                "width": img["width"],
                "height": img["height"],
                "image_id": img["id"],
                "annotations": [anno],
                "sem_seg_file_name": os.path.join(dir_path, img["file_name"][:-4] + "!!.png")
            })
        if len(dataset_dicts):
            log.warning(f"Please double check your custom dict structure to be sure:\n{dataset_dicts[0]['annotations'][0]['bbox']}")

        return dataset_dicts

    def get_hair_dicts_d(dir_path, dir_name="train", dataset_name="large", class_name="hair"):
        dataset_dicts = []
        imgs = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]
        for idx in tqdm(range(len(imgs)), "Accessing json annotation: "):
            img_file = imgs[idx]
            img_path = os.path.join(dir_path, img_file)
            mask_path = img_path[:-4] + "!!.png"
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            bbox = cv2.boundingRect(cv2.findNonZero(mask))

            # only one object per image
            objs = [{
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": Mask(mask).polygons().segmentation,
                "category_id": 0,
                "image_id": idx,
                "id": idx
            }]

            dataset_dicts.append({
                "file_name": img_path,
                "width": mask.shape[1],
                "height": mask.shape[0],
                "image_id": idx,
                "annotations": objs,
                "sem_seg_file_name": mask_path
            })

        return dataset_dicts

    def get_hair_dicts_f(dir_path, dir_name="train", dataset_name="large", class_name="hair"):
        name_prefix = dataset_name + "_" + class_name + "_" + dir_name
        json_file = name_prefix + ".json"
        json_path = os.path.join(dir_path, json_file)
        with open(json_path, "r") as f:
            print(f"loading {json_path}")
            import rapidjson
            json_dict = rapidjson.load(f)
            print(f"loaded {json_path}")

        for idx in range(len(json_dict)):
            # del json_dict[idx]["sem_seg_file_name"]
            # json_dict[idx]["annotations"][0]["bbox"] = np.array(json_dict[idx]["annotations"][0]["bbox"])
            # del json_dict[idx]["annotations"][0]["bbox_mode"]
            # json_dict[idx]["annotations"][0]["is_crowd"] = False
            json_dict[idx]["annotations"][0]["bbox_mode"] = BoxMode.XYWH_ABS
            # json_dict[idx]["file_name"] = os.path.basename(json_dict[idx]["file_name"])
            # json_dict[idx]["file_name"] = os.path.join(dir_path, os.path.basename(json_dict[idx]["file_name"]))

            segm = json_dict[idx]["annotations"][0]["segmentation"]
            if len(segm) == 0:
                del json_dict[idx]["annotations"][0]
                continue
                print("Deleting empty polygon")
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        del json_dict[idx]["annotations"][0]
                        continue
                        print("Deleting empty polygon")
                json_dict[idx]["annotations"][0]["segmentation"] = segm
        return json_dict

    try:
        dataset_dir = os.environ["DETECTRON2_DATASETS"]
    except KeyError:
        dataset_dir = "./datasets"
    class_name = "hair"
    # for dataset_name in ["trail", "large"]:
    for dataset_name in ["large"]:
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        for d in ["val", "train"]:
            name_prefix = dataset_name + "_" + class_name + "_" + d
            type_path = os.path.join(dataset_dir, d)
            # json_file = name_prefix + ".json"
            # json_path = os.path.join(type_path, json_file)
            # register_coco_instances(name_prefix, {}, json_path, type_path)
            # DatasetCatalog.register(name_prefix, get_hair_dicts(type_path, d, dataset_name, class_name))
            from functools import partial
            loader = partial(get_hair_dicts_f, type_path, d, dataset_name, class_name)
            DatasetCatalog.register(name_prefix, loader)
            MetadataCatalog.get(name_prefix).set(thing_classes=[class_name])


if __name__ == "__main__":
    # python train_net.py --config-file configs/InstanceSegmentation/hair_net_trail.yaml --num-gpus 1 --resume
    # python train_net.py --config-file configs/SemanticSegmentation/hair_net_seg.yaml --num-gpus 4
    # export DETECTRON2_DATASETS=~/labnote/datasets
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
