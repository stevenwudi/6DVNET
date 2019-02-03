import torch
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes
from .box_trans_feature_extractor import make_box_trans_feature_extractor
from .trans_predictor import make_roi_trans_predictor
from .loss import make_roi_trans_evaluator


class ROITransHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_box_trans_feature_extractor(cfg)
        self.predictor = make_roi_trans_predictor(cfg)
        self.loss_evaluator = make_roi_trans_evaluator(cfg)

    def forward(self, box_features, x_car_cls_rot, proposals, targets=None):

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        x_mlp = self.feature_extractor(box_features, proposals)
        trans_pred = self.predictor(x_mlp, x_car_cls_rot)

        if not self.training:
            raise NotImplementedError

        loss_type = self.MODEL.TRANS_HEAD.TRANS_LOSS
        loss_trans = self.loss_evaluator(proposals, trans_pred, targets, loss_type)

        return x_mlp, all_proposals, dict(loss_tran=loss_trans)


def build_trans_head(cfg):
    return ROITransHead(cfg)

