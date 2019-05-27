import torch
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


class CarClsRotLoss(object):
    def __init__(self, proposal_matcher, cfg):
        """
        Arguments:
        proposal_matcher (Matcher)
        """
        self.proposal_matcher = proposal_matcher
        self.cfg = cfg.clone()
        if self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.ROT_DIFF_DEGREE:
            # a similarity matrix specific to ApolloScape 3D car
            self.shape_sim_mat = np.loadtxt('../maskrcnn_benchmark/data/datasets/evaluation/apollo_3d_car_instace/sim_mat.txt')

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(self.cfg['MODEL']['ROI_CAR_CLS_ROT_HEAD']['REGRESS_TARGET'])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        car_cat_classes = []
        quaternions = []
        labels = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            quaternion = matched_targets.get_field(self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.ROT_NAME)
            quaternion = quaternion[positive_inds]

            car_cat_classe = matched_targets.get_field(self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.SUBCLASS_NAME)
            car_cat_classe = car_cat_classe[positive_inds]

            labels.append(labels_per_image)
            car_cat_classes.append(car_cat_classe)
            quaternions.append(quaternion)

        return labels, car_cat_classes, quaternions

    def __call__(self, proposals, cls_score, rot_pred, targets, loss_type, ce_weight):
        """

        :param proposals: (list[BoxList])
        :param cls_score: score for car car classification
        :param rot_pred: rotation prediction (quaternions)
        :param targets: (list[BoxList])
        :param loss_type: loss type for rotation prediction ['L1', 'MSE', 'ARCCOS', 'HUBER']
        :param ce_weight: cross entropy weight to balance out the infrequent car classes
        :return:
        """
        labels, car_cat_classes, quaternions = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        car_cat_classes = cat(car_cat_classes, dim=0)
        quaternions = cat(quaternions, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        device_id = cls_score.get_device()

        if len(ce_weight):
            ce_weight = Variable(torch.from_numpy(np.array(ce_weight)).float()).cuda(device_id)
            loss_cls = F.cross_entropy(cls_score[positive_inds], car_cat_classes[positive_inds], ce_weight)
        else:
            loss_cls = F.cross_entropy(cls_score[positive_inds], car_cat_classes[positive_inds])

        # class accuracy
        cls_preds = cls_score.max(dim=1)[1].type_as(car_cat_classes)
        accuracy_cls = cls_preds.eq(car_cat_classes).float().mean(dim=0)

        quaternions = quaternions.type_as(rot_pred)
        # loss rot
        if loss_type == 'L1':
            loss_rot = torch.abs(rot_pred - quaternions)
            N = loss_rot.size(0)  # batch size
            loss_rot = loss_rot.view(-1).sum(0) / N
        elif loss_type == 'MSE':
            loss_rot = (rot_pred - quaternions) ** 2
            N = loss_rot.size(0)  # batch size
            loss_rot = loss_rot.view(-1).sum(0) / N
        elif loss_type == 'ARCCOS':
            pi = Variable(torch.tensor([np.pi]).to(torch.float32)).cuda(device_id)
            diff = torch.abs((rot_pred * quaternions).sum(dim=1))
            loss_rot = 2 * torch.acos(diff) * 180 / pi
            N = diff.size(0)  # batch size
            loss_rot = loss_rot.view(-1).sum(0) / N
        elif loss_type == 'HUBER':
            degree = self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.ROT_HUBER_THRESHOLD
            loss_rot = huber_loss_rot(rot_pred, quaternions, device_id, degree)

        if self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.ROT_DIFF_DEGREE:
            rot_diff_degree = rot_sim_cal(rot_pred.data.cpu().numpy(), quaternions.data.cpu().numpy())
            rot_diff_degree = torch.tensor(rot_diff_degree).type_as(rot_pred).detach()

            # car shape similarity
            shape_sim = shape_sim_cal(cls_preds.data.cpu().numpy(), car_cat_classes.data.cpu().numpy(), self.shape_sim_mat, self.cfg.MODEL.ROI_CAR_CLS_ROT_HEAD.UNIQUE_CAR_MODELS)
            shape_sim = torch.tensor(shape_sim).type_as(rot_pred).detach()

            return loss_cls, loss_rot, accuracy_cls, rot_diff_degree, shape_sim
        else:
            return loss_cls, loss_rot, accuracy_cls


def rot_sim_cal(dt_car_rot, gt_car_rot):
    q1 = dt_car_rot / np.linalg.norm(dt_car_rot, axis=1)[:, None]
    q2 = gt_car_rot / np.linalg.norm(gt_car_rot, axis=1)[:, None]
    diff = abs(np.sum(q1*q2, axis=1))
    dis_rot = 2 * np.arccos(diff) * 180 / np.pi
    dis_rot = dis_rot.mean()

    return dis_rot


def shape_sim_cal(pred_car, car_cls_labels_int32, shape_sim_mat, unique_car_models):
    """

    :param car_cls_prop: N * N_car_classes (34 or 79)
    :param shape_sim_mat: N * N_car_classes (34 or 79)
    :return:
    """

    unique_car_models = np.array(unique_car_models)
    shape_sim_mat_34 = shape_sim_mat[unique_car_models, :][:, unique_car_models]
    shape_sim = shape_sim_mat_34[pred_car, car_cls_labels_int32]

    return shape_sim.mean()


def huber_loss_rot(trans_pred, label_trans, device_id, degree=5):
    """
    beta is re-calculated as:
    np.cos(5./360 * np.pi) = 0.999
    0.999 = abs ( 1- diff**2) --> beta = sqrt(0.001) = -.0316
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    https://en.wikipedia.org/wiki/Huber_loss
    """
    degree_diff = np.cos(degree/360. * np.pi)
    beta = np.sqrt(2*(1 - degree_diff))

    quaternion_diff = trans_pred - label_trans

    pi = Variable(torch.tensor([np.pi]).to(torch.float32)).cuda(device_id)
    diff = torch.abs((trans_pred * label_trans).sum(dim=1))
    degree_diff = 2 * torch.acos(diff) * 180 / pi
    inbox_idx = degree_diff <= degree
    outbox_idx = degree_diff > degree

    bbox_inside_weights = Variable(torch.tensor(inbox_idx, dtype=torch.float32)).cuda(device_id)
    bbox_outside_weights = Variable(torch.tensor(outbox_idx, dtype=torch.float32)).cuda(device_id)

    in_box_pow_diff = 0.5 * torch.pow(quaternion_diff, 2) / beta
    in_box_loss = in_box_pow_diff.sum(dim=1) * bbox_inside_weights

    out_box_abs_diff = torch.abs(quaternion_diff)
    out_box_loss = (out_box_abs_diff.sum(dim=1) - beta / 2) * bbox_outside_weights

    loss_box = in_box_loss + out_box_loss
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def make_roi_car_cls_rot_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = CarClsRotLoss(matcher, cfg)

    return loss_evaluator

