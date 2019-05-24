from torch import nn
import torch.nn.functional as F


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        representation_size = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.MLP_HEAD_DIM
        num_car_classes = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.NUMBER_CARS
        num_rot = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.NUMBER_ROTS

        self.cls_score = nn.Linear(representation_size, num_car_classes)
        self.rot_pred = nn.Linear(representation_size, num_rot)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.rot_pred.weight, std=0.001)
        for l in [self.cls_score, self.rot_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        cls_score = self.cls_score(x)
        cls = F.softmax(cls_score, dim=1)

        rot_pred = self.rot_pred(x)
        rot_pred = F.normalize(rot_pred, p=2, dim=1)

        return cls_score, cls, rot_pred


_ROI_CAR_CLS_ROT_PREDICTOR = {"FPNPredictor": FPNPredictor}


def make_roi_car_cls_rot_predictor(cfg):
    func = _ROI_CAR_CLS_ROT_PREDICTOR[cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.PREDICTOR]
    return func(cfg)