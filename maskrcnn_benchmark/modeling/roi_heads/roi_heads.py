# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .car_cls_rot_head.car_cls_rot_head import build_roi_car_cls_rot_head
from .trans_head.trans_head import build_trans_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.CAR_CLS_HEAD_ON and cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.car_cls_rot.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        if not self.cfg.MODEL.TRANS_HEAD_ON:
            x, detections, loss_box = self.box(features, proposals, targets)
        else:
            x, det_result, detections, loss_box = self.box(features, proposals, targets)

        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.CAR_CLS_HEAD_ON:
            car_cls_rot_features = features
            if self.training and self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            x_car_cls_rot, detections, loss_car_cls = self.car_cls_rot(car_cls_rot_features, detections, targets)
            losses.update(loss_car_cls)

            if self.cfg.MODEL.TRANS_HEAD_ON:
                trans_pred, loss_trans = self.trans(x_car_cls_rot, det_result, detections, targets)
                losses.update(loss_trans)

        return x, detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))
    if cfg.MODEL.CAR_CLS_HEAD_ON:
        roi_heads.append(("car_cls_rot", build_roi_car_cls_rot_head(cfg)))
        # We assume the feature for translational head will depend upon the car cls rot head
        if cfg.MODEL.TRANS_HEAD_ON:
            roi_heads.append(("trans", build_trans_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
