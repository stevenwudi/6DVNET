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

    def forward(self, x_car_cls_rot, det_result, proposals, targets=None):

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        # We use ground truth box to feed translation head MLP
        # TODO: use curriculum learning method that incorporate model bbox info.
        device = x_car_cls_rot.device
        box_features = self.bbox_transform_by_intrinsics(proposals, device)

        x_mlp = self.feature_extractor(box_features)
        trans_pred = self.predictor(x_mlp, x_car_cls_rot)

        if not self.training:
            raise NotImplementedError

        loss_type = self.cfg.MODEL.TRANS_HEAD.TRANS_LOSS
        loss_trans = self.loss_evaluator(proposals, trans_pred, targets, loss_type)
        loss_trans *= self.cfg.MODEL.TRANS_HEAD.TRANS_LOSS_BETA

        return x_mlp, dict(loss_tran=loss_trans)

    def bbox_transform_by_intrinsics(self, proposals, device):

        pred_boxes = torch.zeros(proposals[0].bbox.shape, dtype=proposals[0].bbox.dtype, device=device)

        # x_c
        pred_boxes[:, 0::4] = (proposals[0].bbox[:, 0::4] + proposals[0].bbox[:, 0::4])/2
        # y_c
        pred_boxes[:, 1::4] = (proposals[0].bbox[:, 1::4] + proposals[0].bbox[:, 3::4])/2
        # w (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = proposals[0].bbox[:, 2::4] - proposals[0].bbox[:, 0::4]
        # h (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = proposals[0].bbox[:, 3::4] - proposals[0].bbox[:, 1::4]

        intrinsic_vect = self.cfg.MODEL.TRANS_HEAD.CAMERA_INTRINSIC
        pred_boxes[:, 0::4] -= intrinsic_vect[2]
        pred_boxes[:, 0::4] /= intrinsic_vect[0]
        pred_boxes[:, 1::4] -= intrinsic_vect[3]
        pred_boxes[:, 1::4] /= intrinsic_vect[1]

        pred_boxes[:, 2::4] /= intrinsic_vect[0]
        pred_boxes[:, 3::4] /= intrinsic_vect[1]

        return pred_boxes


def build_trans_head(cfg):
    return ROITransHead(cfg)

