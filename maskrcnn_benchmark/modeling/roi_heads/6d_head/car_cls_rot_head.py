import torch

from .roi_car_cls_rot_feature_extractors import make_roi_car_cls_rot_feature_extractor



class ROICarClsRotHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.feature_extactor = make_roi_car_cls_rot_feature_extractor(cfg)
        self.predictor = make_roi_car_cls_rot_predictor(cfg)


def build_car_cls_rot_head(cfg):
    return ROICarClsRotHead(cfg)