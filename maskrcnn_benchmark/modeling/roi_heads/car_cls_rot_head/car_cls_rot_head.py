import torch
from .roi_car_cls_rot_feature_extractors import make_roi_car_cls_rot_feature_extractor
from .roi_car_cls_rot_predictor import make_roi_car_cls_rot_predictor
from .loss import make_roi_car_cls_rot_evaluator
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes


class ROICarClsRotHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_car_cls_rot_feature_extractor(cfg)
        self.predictor = make_roi_car_cls_rot_predictor(cfg)
        self.loss_evaluator = make_roi_car_cls_rot_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments
        :param features: (list[Tensor]): feature-maps from possibly several levels
        :param proposals:  (list[BoxList]): proposal boxes
        :param targets: (list[BoxList], optional): the ground-truth targets.
        :return:
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        cls_score, cls, rot_pred = self.predictor(x)

        if not self.training:
            assert NotImplementedError

        loss_type = self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.ROT_LOSS
        ce_weight = self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.CE_CAR_CLS_FINETUNE_WIGHT
        loss_cls, loss_rot, accuracy_cls = self.loss_evaluator(proposals, cls_score, rot_pred, targets, loss_type, ce_weight)

        return x, all_proposals, dict(loss_car_cls=loss_cls, loss_rot=loss_rot, accuracy_car_cls=accuracy_cls)


def build_roi_car_cls_rot_head(cfg):
    return ROICarClsRotHead(cfg)