import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def huber_loss(bbox_pred, bbox_targets, device_id, beta=2.8):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    https://en.wikipedia.org/wiki/Huber_loss
    """
    box_diff = bbox_pred - bbox_targets

    dis_trans = np.linalg.norm(box_diff.data.cpu().numpy(), axis=1)
    # we also add a metric for dist<2.8 metres.
    inbox_idx = dis_trans <= 2.8
    outbox_idx = dis_trans > 2.8

    bbox_inside_weights = Variable(torch.from_numpy(inbox_idx.astype('float32'))).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(outbox_idx.astype('float32'))).cuda(device_id)

    in_box_pow_diff = 0.5 * torch.pow(box_diff, 2) / beta
    in_box_loss = in_box_pow_diff.sum(dim=1) * bbox_inside_weights

    out_box_abs_diff = torch.abs(box_diff)
    out_box_loss = (out_box_abs_diff.sum(dim=1) - beta / 2) * bbox_outside_weights

    loss_box = in_box_loss + out_box_loss
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


class TransLoss(object):
    def __init__(self, proposal_matcher, cfg):
        """
        Arguments:
        proposal_matcher (Matcher)
        """
        self.proposal_matcher = proposal_matcher
        self.cfg = cfg.clone()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Trans head needs "labels" and "poses "fields for creating the targets
        target = target.copy_with_fields(["labels", "poses"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        trans_labels_list = []
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

            label_trans = matched_targets.get_field("quaternions")
            label_trans = label_trans[positive_inds]

            labels.append(labels_per_image)
            trans_labels_list.append(label_trans)

        return labels, trans_labels_list

    def __call__(self, proposals, trans_pred, targets, loss_type):
        """

        :param proposals: (list[BoxList])
        :param trans_pred:
        :param targets:(list[BoxList])
        :return:
        """
        labels, label_trans = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        label_trans = cat(label_trans, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        device_id = label_trans.get_device()

        if loss_type == 'MSE':
            loss = nn.MSELoss()
            loss_trans = loss(trans_pred, label_trans[positive_inds])
        elif loss_type == 'L1':
            loss = nn.L1Loss()
            loss_trans = loss(trans_pred, label_trans[positive_inds])
        elif loss_type == 'HUBER':
            beta = self.cfg.MODEL.TRANS_HEAD.TRANS_HUBER_THRESHOLD
            loss_trans = huber_loss(trans_pred, label_trans[positive_inds], device_id, beta)

        return loss_trans


def make_roi_trans_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = TransLoss(matcher, cfg)

    return loss_evaluator