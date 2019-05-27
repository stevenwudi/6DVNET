import torch
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes
from .box_trans_feature_extractor import make_box_trans_feature_extractor
from .trans_predictor import make_roi_trans_predictor
from .loss import make_roi_trans_evaluator
from .inference import make_roi_trans_post_processor
from maskrcnn_benchmark.modeling.utils import cat


class ROITransHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_box_trans_feature_extractor(cfg)
        self.predictor = make_roi_trans_predictor(cfg)
        self.post_processor = make_roi_trans_post_processor(cfg)
        self.loss_evaluator = make_roi_trans_evaluator(cfg)

    def forward(self, x_car_cls_rot, det_result, proposals, targets=None):

        if self.training:
            # during training, only focus on positive boxes
            all_propoals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        # We use ground truth box to feed translation head MLP
        # TODO: use curriculum learning method that incorporate model bbox info.
        device = x_car_cls_rot.device

        concat_boxes = cat([b.bbox for b in proposals], dim=0)

        # get the img scale
        im_scale = targets[0].get_field('im_scales')[0]
        box_features = self.bbox_transform_by_intrinsics(concat_boxes, im_scale, device)

        x_mlp = self.feature_extractor(box_features)
        trans_pred = self.predictor(x_mlp, x_car_cls_rot)

        if not self.training:
            result = self.post_processor(trans_pred, proposals)
            return result, {}

        loss_type = self.cfg.MODEL.TRANS_HEAD.TRANS_LOSS
        loss_trans, dis_trans = self.loss_evaluator(proposals, trans_pred, targets, loss_type)
        loss_trans *= self.cfg.MODEL.TRANS_HEAD.TRANS_LOSS_BETA

        return trans_pred, dict(loss_tran=loss_trans, dis_trans=dis_trans)

    def bbox_transform_by_intrinsics(self, concat_boxes, im_scale, device):

        pred_boxes = torch.zeros(concat_boxes.shape, dtype=concat_boxes.dtype, device=device)

        # x_c
        pred_boxes[:, 0::4] = (concat_boxes[:, 0::4] + concat_boxes[:, 2::4])/2
        # y_c
        pred_boxes[:, 1::4] = (concat_boxes[:, 1::4] + concat_boxes[:, 3::4])/2
        # w (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = concat_boxes[:, 2::4] - concat_boxes[:, 0::4]
        # h (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = concat_boxes[:, 3::4] - concat_boxes[:, 1::4]

        intrinsic_vect = np.array(self.cfg.MODEL.TRANS_HEAD.CAMERA_INTRINSIC)
        if self.cfg.INPUT.BOTTOM_HALF:
            intrinsic_vect[3] /= 2

        pred_boxes[:, 0::4] -= intrinsic_vect[2]
        pred_boxes[:, 0::4] /= intrinsic_vect[0]
        pred_boxes[:, 1::4] -= intrinsic_vect[3]
        pred_boxes[:, 1::4] /= intrinsic_vect[1]

        pred_boxes[:, 2::4] /= intrinsic_vect[0]
        pred_boxes[:, 3::4] /= intrinsic_vect[1]

        pred_boxes /= im_scale

        return pred_boxes


def build_trans_head(cfg):
    return ROITransHead(cfg)

