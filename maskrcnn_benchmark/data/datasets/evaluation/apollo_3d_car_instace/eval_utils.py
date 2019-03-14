"""
    Brief: Compute similarity metrics for evaluation
    Author: wangpeng54@baidu.com
    Date: 2018/6/20
"""

import numpy as np
import math
from maskrcnn_benchmark.data.datasets.evaluation.apollo_3d_car_instace import utils as uts


def pose_similarity(dt, gt, shape_sim_mat):
    """compute pose similarity in terms of shape, translation and rotation
    Input:
        dt: np.ndarray detection [N x 7] first 6 dims are roll, pitch, yaw, x, y, z
        gt: save with dt

    Output:
        sim_shape: similarity based on shape eval
        dis_trans: distance based on translation eval
        dis_rot:   dis.. based on rotation eval
    """
    dt_num = len(dt)
    gt_num = len(gt)
    car_num = shape_sim_mat.shape[0]

    dt_car_id = np.uint32(dt[:, -1])
    gt_car_id = np.uint32(gt[:, -1])

    idx = np.tile(dt_car_id[:, None], (1, gt_num)).flatten() * car_num + \
          np.tile(gt_car_id[None, :], (dt_num, 1)).flatten()
    sims_shape = shape_sim_mat.flatten()[idx]
    sims_shape = np.reshape(sims_shape, [dt_num, gt_num])

    # translation similarity
    dt_car_trans = dt[:, 3:-1]
    gt_car_trans = gt[:, 3:-1]
    dis_trans = np.linalg.norm(np.tile(dt_car_trans[:, None, :], [1, gt_num, 1]) -
                               np.tile(gt_car_trans[None, :, :], [dt_num, 1, 1]), axis=2)

    # rotation similarity
    dt_car_rot = uts.euler_angles_to_quaternions(dt[:, :3])
    gt_car_rot = uts.euler_angles_to_quaternions(gt[:, :3])
    q1 = dt_car_rot / np.linalg.norm(dt_car_rot, axis=1)[:, None]
    q2 = gt_car_rot / np.linalg.norm(gt_car_rot, axis=1)[:, None]

    # diff = abs(np.matmul(q1, np.transpose(q2)))
    diff = abs(1 - np.sum(np.square(np.tile(q1[:, None, :], [1, gt_num, 1]) - np.tile(q2[None, :, :], [dt_num, 1, 1])), axis=2) / 2.0)
    dis_rot = 2 * np.arccos(diff) * 180 / np.pi

    return sims_shape, dis_trans, dis_rot


def shape_sim(car_cls_prop, shape_sim_mat, car_cls_labels_int32):
    """

    :param car_cls_prop: N * N_car_classes (34 or 79)
    :param shape_sim_mat: N * N_car_classes (34 or 79)
    :return:
    """
    if cfg.CAR_CLS.SIM_MAT_LOSS:
        pred_car = np.argmax(car_cls_prop, axis=1)
        shape_sim= shape_sim_mat[car_cls_labels_int32, pred_car]

    else:
        unique_car_models = np.array(cfg.TRAIN.CAR_MODELS)
        shape_sim_mat_34 = shape_sim_mat[unique_car_models, :][:, unique_car_models]
        pred_car = np.argmax(car_cls_prop, axis=1)
        shape_sim= shape_sim_mat_34[pred_car, car_cls_labels_int32]

    return shape_sim.mean()


def rot_sim(dt_car_rot, gt_car_rot):
    q1 = dt_car_rot / np.linalg.norm(dt_car_rot, axis=1)[:, None]
    q2 = gt_car_rot / np.linalg.norm(gt_car_rot, axis=1)[:, None]

    diff = abs(np.sum(q1*q2, axis=1))
    dis_rot = 2 * np.arccos(diff) * 180 / np.pi
    return dis_rot.mean()


def trans_sim(dt_car_trans, gt_car_trans, mean, std):
    # translation similarity
    mean = np.array(mean)
    std = np.array(std)
    if cfg.TRANS_HEAD.NORMALISE:
        dt_car_trans = dt_car_trans * std + mean
        gt_car_trans = gt_car_trans * std + mean
    dis_trans = np.linalg.norm(dt_car_trans-gt_car_trans, axis=1)

    # we also add a metric for dist<2.8 metres.
    trans_thresh_per = np.sum(dis_trans < 2.8) / dis_trans.shape[0]
    return dis_trans.mean(), trans_thresh_per


def IOU(mask1, mask2):
    """ compute the intersection of union of two logical inputs
    Input:
        mask1: the first mask
        mask2: the second mask
    """

    inter = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    if np.sum(inter) == 0:
        return 0.

    return np.float32(np.sum(inter)) / np.float32(np.sum(union))


def quaternion_to_euler_angle(q):

    """Convert quaternion to euler angel.
    Input:
        q: 1 * 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [roll, pitch, yaw]
    """
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

if __name__ == '__main__':
    shape_sim_mat = np.loadtxt('./test_eval_data/sim_mat.txt')
    fake_gt = []
    fake_dt = []
    sim_shape, sim_trans, sim_rot = pose_similarity(fake_dt, fake_gt, shape_sim_mat)
