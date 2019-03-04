# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


# object
class TransPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, cfg):
        super(TransPostProcessor, self).__init__()
        self.cfg = cfg.clone()

    def forward(self, trans_pred, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        boxes_per_image = [len(box) for box in boxes]
        trans_pred = trans_pred.split(boxes_per_image, dim=0)

        results = []
        for trans, box in zip(trans_pred, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("trans_pred", trans)
            results.append(bbox)

        return results


def make_roi_trans_post_processor(cfg):
    trans_post_processor = TransPostProcessor(cfg)
    return trans_post_processor
