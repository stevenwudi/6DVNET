# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


# object
class CarClsRotPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, cfg):
        super(CarClsRotPostProcessor, self).__init__()
        self.cfg = cfg.clone()

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        cls_score, cls, rot_pred = x['cls_score'], x['cls'], x['rot_pred']
        num_cars = cls_score.shape[0]

        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_cars, device=labels.device)

        cls_label = torch.argmax(cls, dim=1)
        cls_score = cls[index, cls_label][:, None]

        boxes_per_image = [len(box) for box in boxes]
        cls_label = cls_label.split(boxes_per_image, dim=0)
        cls_score = cls_score.split(boxes_per_image, dim=0)
        rot_pred = rot_pred.split(boxes_per_image, dim=0)

        results = []
        for label, score, rot, box in zip(cls_label, cls_score, rot_pred, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("cls_score", score)
            bbox.add_field("cls", label)
            bbox.add_field("rot_pred", rot)
            results.append(bbox)

        return results


def make_roi_car_cls_rot_post_processor(cfg):
    car_cls_rot_post_processor = CarClsRotPostProcessor(cfg)
    return car_cls_rot_post_processor
