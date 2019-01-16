# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util
from maskrcnn_benchmark.data.datasets.evaluation.kitti.kittieval import KITTIeval
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults, check_expected_results


def do_kitti_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):

    logger = logging.getLogger("maskrcnn_benchmark.inference")

    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(pred_boxlists, gt_boxlists, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return

    kitti_results = {}
    kitti_gts = {}

    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        kitti_results["bbox"] = prepare_for_kitti_detection(pred_boxlists)
        kitti_gts["bbox"] = prepare_for_kitti_detection(pred_boxlists)

    if "segm" in iou_types:
        logger.info("Preparing segm results")
        kitti_results["segm"] = prepare_for_kitti_segmentation(pred_boxlists)
        kitti_gts["segm"] = prepare_for_kitti_segmentation(pred_boxlists)

    #results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        res = evaluate_predictions_on_kitti(kitti_gts, kitti_results, iou_type)
    #     results.update(res)
    # logger.info(results)
    #check_expected_results(results, expected_results, expected_results_sigma_tol)
    #
    #return results
    return res


def evaluate_box_proposals(predictions, gt_boxlists, thresholds=None, area="all", limit=None):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):

        gt_boxes = gt_boxlists[image_id]
        gt_areas = torch.as_tensor(gt_boxes.area())
        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]
        num_pos += len(gt_boxes)
        if len(gt_boxes) == 0:
            continue
        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        prediction = prediction.resize(gt_boxes.size)
        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def prepare_for_kitti_detection(pred_boxlists):
    kitti_results = []
    for image_id, prediction in enumerate(pred_boxlists):
        if len(prediction) == 0:
            continue

        boxes = prediction.bbox.tolist()
        areas = [(b[3]-b[1])*(b[2]-b[0]) for b in boxes]

        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        kitti_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                    'area': areas[k],
                    'id': k + 1,
                }
                for k, box in enumerate(boxes)
            ]
        )
    return kitti_results


def prepare_for_kitti_segmentation(pred_boxlists):

    masker = Masker(threshold=0.5, padding=1)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(pred_boxlists)):
        if len(prediction) == 0:
            continue

        masks = prediction.get_field("mask")
        areas = masks.sum(-1).sum(-1)
        masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
        masks = masks[0]

        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        rles = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0] for mask in masks]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                    'area': areas[k],
                    'id': k+1,
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def evaluate_predictions_on_kitti(kitti_gts, kitti_results, iou_type="bbox"):

    kitti_eval = KITTIeval(kitti_gts, kitti_results, iou_type)
    kitti_eval.evaluate()
    kitti_eval.accumulate()
    kitti_eval.summarize()
    return kitti_eval
