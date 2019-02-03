import torch
from torch import nn
import torch.nn.functional as F


class MLPCONCATPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        representation_size = cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.MLP_HEAD_DIM
        box_representation_classes = cfg.MODEL.TRANS_HEAD.MLP_HEAD_DIM

        self.car_cls_rot_linear = nn.Linear(representation_size, cfg.MODEL.TRANS_HEAD.MLP_HEAD_DIM)
        self.trans_pred = nn.Linear(box_representation_classes + cfg.MODEL.TRANS_HEAD.MLP_HEAD_DIM, cfg.MODEL.TRANS_HEAD.OUTPUT_DIM)

        nn.init.normal_(self.car_cls_rot_linear.weight, std=0.01)
        nn.init.normal_(self.trans_pred.weight, std=0.1)
        for l in [self.car_cls_rot_linear, self.trans_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x_mlp, x_car_cls_rot):
        if x_mlp.dim() == 4:
            x_mlp = x_mlp.squeeze(3).squeeze(2)

        batch_size = x_mlp.size(0)

        x_car_cls_rot = F.relu(self.car_cls_rot_linear(x_car_cls_rot.view(batch_size, -1)), inplace=True)
        x_merge = F.relu(torch.cat((x_mlp, x_car_cls_rot), dim=1))
        trans_pred = self.trans_pred(x_merge)
        return trans_pred


_ROI_TRANS_PREDICTOR = {"MLPCONCATPredictor": MLPCONCATPredictor}


def make_roi_trans_predictor(cfg):
    func = _ROI_TRANS_PREDICTOR[cfg.MODEL.TRANS_HEAD.PREDICTOR]
    return func(cfg)