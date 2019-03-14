"""
    Brief: Evaluation of 3D car instance mean AP following coco eval criteria
    Author: wangpeng54@baidu.com, (Piotr Dollar and Tsung-Yi Lin)
    Date: 2018/6/20
"""

import os
import json
import numpy as np
import datetime
import argparse
import time
import cv2
from maskrcnn_benchmark.data.datasets.evaluation.apollo_3d_car_instace import eval_utils as euts
from maskrcnn_benchmark.data.datasets.evaluation.apollo_3d_car_instace import utils
from maskrcnn_benchmark.utils.colormaps import colormap
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.data.datasets.evaluation.apollo_3d_car_instace.wad_eval import WAD_eval

from collections import defaultdict
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle

Criterion = namedtuple('Criterion', [
    'shapeSim',  # Criterion for shape similarity
    'transDis',  # thresholds for translation
    'oriDis',  # thresholds for orientation
])


def do_3d_car_instance_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    cfg,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    vis=False
):
    # We first write prediction to the required submission json format
    if not len(dataset.img_list_all) == len(predictions):
        raise AssertionError("Prediction is not the same length as images numbers")

    output_json_dir = output_folder +'/json'
    output_vis_dir = output_folder +'/vis'
    # Write to json file
    if not os.path.isdir(output_json_dir):
        os.mkdir(output_json_dir)
    if len(os.listdir(output_json_dir)) < len(dataset.img_list_all):
        write_pose_to_json(dataset, predictions, output_json_dir, thresh=cfg.TEST.SCORE_THRESH_FOR_TRUTH_DETECTION, iou_ignore_threshold=False)
    # visualise the image, we have such a bad visualisation result
    if vis:
        if not os.path.isdir(output_vis_dir):
            os.mkdir(output_vis_dir)
        if len(os.listdir(output_vis_dir)) < len(dataset.img_list_all):
            vis_image_eccv2018_car_3d(dataset, predictions, output_vis_dir, thresh=cfg.TEST.SCORE_THRESH_FOR_TRUTH_DETECTION, dpi=200, box_alpha=0.0, ext='jpg')

    # We First evaluate the box, segm and then evaluate the 3D metric
    # First boxes
    eval_type = 'boxes'
    gt_file = os.path.join(output_folder, 'val_gt_' + eval_type + '.pth')
    # Since reading GT need to retrieve from the hard disc, so we try to cache it.
    if os.path.isfile(gt_file):
        coco_gt = pickle.load(open(gt_file, "rb"))
    else:
        coco_gt = dataset.loadGt(type=eval_type)
        with open(gt_file, "wb") as f:
            pickle.dump(coco_gt, f)
        pickle.dump(coco_gt, open(gt_file, "wb"))

    coco_dt = dataset.loadRes(predictions, type=eval_type)

    coco_eval = WAD_eval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Now we evaluate the segm
    eval_type = 'segms'
    gt_file = os.path.join(output_folder, 'val_gt_' + eval_type + '.pth')
    # Since reading GT need to retrieve from the hard disc, so we try to cache it.
    if os.path.isfile(gt_file):
        coco_gt = pickle.load(open(gt_file, "rb"))
    else:
        coco_gt = dataset.loadGt(type=eval_type)
        pickle.dump(coco_gt, open(gt_file, "wb"))

    dt_file = os.path.join(output_folder, 'val_dt_' + eval_type + '.pth')
    if os.path.isfile(gt_file):
        coco_dt = pickle.load(open(dt_file, "rb"))
    else:
        coco_dt = dataset.loadRes(predictions, type=eval_type)
        pickle.dump(coco_dt, open(dt_file, "wb"))

    coco_eval = WAD_eval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()

    # The following evaluate the mAP of car poses
    args.test_dir = output_json_dir
    args.gt_dir = args.dataset_dir + 'car_poses'
    args.res_file = os.path.join(output_dir, 'json_'+args.list_flag+'_res.txt')
    args.simType = None
    det_3d_metric = Detect3DEval(args)
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


def write_pose_to_json(dataset, predictions, output_dir, thresh, iou_ignore_threshold=True):
    """
    Write a Json file for submission
    """
    for im_name, prediction in zip(dataset.img_list_all, predictions):
        json_file = os.path.join(output_dir, im_name+'.json')
        # Get the field from BoxList
        boxes = prediction.bbox
        car_cls = prediction.get_field('cls')
        bbox_score = prediction.get_field('scores')
        masks = prediction.get_field('mask')
        trans_pred = prediction.get_field('trans_pred').numpy()

        quaternions = prediction.get_field('rot_pred').numpy()
        # The following two lines are not necessary if our network output already normalises the output
        norm = np.linalg.norm(quaternions, axis=1)
        rot_pred_norm = quaternions / norm[:, None]
        # normalise the unit quaternion here
        euler_angle = np.array([euts.quaternion_to_euler_angle(x) for x in rot_pred_norm])

        if boxes is None or boxes.shape[0] == 0 or max(bbox_score) < thresh:
            return

        # From largest to smallest order
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

        car_list = []
        for i in sorted_inds:
            score = bbox_score[i]
            if score < thresh:
                continue

            # json_id = dataset.contiguous_category_id_to_json_id[classes[i]]
            # class_string = dataset.id_map_to_cat[json_id]
            class_string = "car"

            # car_cls
            car_cls_i = car_cls[i]
            euler_angle_i = euler_angle[i]
            trans_pred_i = trans_pred[i]
            car_model_i = dataset.unique_car_models[car_cls_i]

            if class_string == 'car':
                # filter out by ignored_mask_binary
                car_info = dict()
                car_info["car_id"] = int(car_model_i)
                car_info["pose"] = [float(x) for x in euler_angle_i] + [float(x) for x in trans_pred_i]
                # We use rectangle area
                car_info["area"] = int(areas[i])
                car_info["score"] = float(score)
                if iou_ignore_threshold:
                    ignored_mask_img = dataset.dataset_dir + '/ignore_mask/' + im_name + '.jpg'
                    ignored_mask = cv2.imread(ignored_mask_img, cv2.IMREAD_GRAYSCALE)
                    ignored_mask_binary = np.zeros(ignored_mask.shape)
                    ignored_mask_binary[ignored_mask > 250] = 1

                    masks = np.zeros_like(ignored_mask_binary)
                    masks[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]): int(boxes[i][2])] = 1
                    iou_mask = masks * ignored_mask_binary
                    iou = np.sum(iou_mask) / int(areas[i])
                    if iou <= iou_ignore_threshold:
                        car_list.append(car_info)
                    else:
                        print('This mask has been ignored')
                else:
                    car_list.append(car_info)

        with open(json_file, 'w') as outfile:
            json.dump(car_list, outfile, indent=4)


def vis_image_eccv2018_car_3d(
        dataset, predictions, output_dir, thresh=0.1,
        dpi=200, box_alpha=0.0, ext='jpg'):

    """Visual debugging of detections."""
    intrinsic = dataset.get_intrinsic_mat()

    masker = Masker(threshold=0.5, padding=1)
    color_list = colormap(rgb=True) / 255
    w_ratio = .4
    for im_name, prediction in zip(dataset.img_list_all, predictions):
        # Get the field from BoxList
        boxes = prediction.bbox
        car_cls = prediction.get_field('cls')
        bbox_score = prediction.get_field('scores')
        masks = prediction.get_field('mask')
        masks = masker([masks], [prediction])[0]
        masks = masks.numpy()
        trans_pred = prediction.get_field('trans_pred').numpy()
        quaternions = prediction.get_field('rot_pred').numpy()
        # The following two lines are not necessary if our network output already normalises the output
        norm = np.linalg.norm(quaternions, axis=1)
        rot_pred_norm = quaternions / norm[:, None]
        # normalise the unit quaternion here
        euler_angle = np.array([euts.quaternion_to_euler_angle(x) for x in rot_pred_norm])

        im = cv2.imread(dataset.dataset_dir + '/images/' + im_name + '.jpg')
        im = im[:, :, ::-1]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        mask_color_id = 0
        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = bbox_score[i]

            if score < thresh:
                continue

            # car_cls
            car_cls_i = car_cls[i]
            euler_angle_i = euler_angle[i]
            trans_pred_i = trans_pred[i]
            car_model_i = dataset.unique_car_models[car_cls_i]
            car_model_name_i = dataset.car_id2name[car_model_i]

            # show box (off by default, box_alpha=0.0)
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                       bbox[2] - bbox[0],
                                       bbox[3] - bbox[1],
                                       fill=False, edgecolor='g',
                                       linewidth=0.5, alpha=box_alpha))

            vis_string = car_model_name_i.name
            ax.text(bbox[0], bbox[1] - 2,
                    vis_string,
                    fontsize=10,
                    family='serif',
                    bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                    color='white')

            # Show predicted pos mesh here:
            #if False:
            if trans_pred_i is not None and euler_angle_i is not None:
                pose = np.concatenate((euler_angle_i, trans_pred_i))
                car_name = car_model_name_i.name
                car = dataset.car_models[car_name]
                pose = np.array(pose)
                # project 3D points to 2d image plane
                rmat = utils.euler_angles_to_rotation_matrix(pose[:3])
                rvect, _ = cv2.Rodrigues(rmat)
                imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], intrinsic, distCoeffs=None)
                triangles = np.array(car['faces']-1).astype('int64')
                x = np.squeeze(imgpts[:, :, 1])
                y = np.squeeze(imgpts[:, :, 0])
                triangles = triangles

                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                for c in range(3):
                    color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio

                ax.triplot(y, x, triangles, alpha=0.5, linewidth=1.2, color=color_mask)

            # show mask
            if False:
                for mask in masks:
                    color_mask = color_list[mask_color_id % len(color_list), 0:3]
                    for c in range(3):
                        color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio

                    e = mask[0, :, :, None]
                    _, contours, hierarchy = cv2.findContours(e.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        polygon = Polygon(
                            c.reshape((-1, 2)),
                            fill=True, facecolor=color_mask,
                            edgecolor='w', linewidth=1.2,
                            alpha=0.05)
                        ax.add_patch(polygon)

            mask_color_id += 1

        output_name = os.path.basename(im_name) + '.' + ext
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
        plt.close('all')


class Detect3DEval(object):
    # Interface for evaluating detection on the Apolloscape 3d car understanding
    #
    # The usage for Detection3DEval is as follows:
    #  E = Detect3DEval(args);      # initialize object
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  image_names     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  shapeThrs  - [.5:.05:.95] T=10 shape thresholds for evaluation
    #  rotThrs    - [50:  5:  5] T=10 rot thresholds for evaluation
    #  transThrs  - [0.1:.3:2.8] T=10 trans thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting

    def __init__(self, args):
        """ Initialize CocoEval using coco APIs for gt and dt
        Input:
            args: configeration object containing test folder and gt folder
        """
        if not args.simType:
            args.simType = '3dpose'
            print('simType not specified. use default simType ')

        self.args = args
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.evalRes = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.sims = {}  # sims between all gts and dts

        #self.image_list = self._checker(args.test_dir, args.gt_dir)
        self.image_list = self._checker_intersection(args.test_dir, args.gt_dir)
        self.params = Params(simType=args.simType)

    def _checker(self, res_folder, gt_folder):
        """Check whether results folder contain same image number
        """
        gt_list = sorted(os.listdir(gt_folder))
        res_list = sorted(os.listdir(res_folder))
        if len(gt_list) != len(res_list):
            raise Exception('results folder image num is not the same as ground truth')

        return gt_list

    def _checker_intersection(self, res_folder, gt_folder):
        """Check whether results folder contain same image number
        """
        gt_list = sorted(os.listdir(gt_folder))
        res_list = sorted(os.listdir(res_folder))
        image_list = list(set(gt_list).intersection(res_list))
        print("GT: %d, RES: %d, Evaluated: %d" % (len(gt_list), len(res_list), len(image_list)))
        return image_list

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        self._gts = {}  # gt for evaluation
        self._dts = {}  # dt for evaluation
        print('loading results')
        count_gt = 1
        count_dt = 1
        for image_name in self.image_list:
            gt_file = '%s/%s' % (self.args.gt_dir, image_name)
            dt_file = '%s/%s' % (self.args.test_dir, image_name)
            car_poses_gt = json.load(open(gt_file, 'r'))
            car_poses_dt = json.load(open(dt_file, 'r'))
            for car_pose in car_poses_gt:
                car_pose['id'] = count_gt
                car_pose['ignore'] = 0
                count_gt += 1

            for car_pose in car_poses_dt:
                car_pose['id'] = count_dt
                count_dt += 1

            self._gts[image_name] = car_poses_gt
            self._dts[image_name] = car_poses_dt

        self.params.image_names = self.image_list
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.evalRes = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        print('Evaluate annotation type *{}*'.format(p.simType))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        if p.simType == '3dpose':
            compute_sim = self.compute_sim
        self.sims = {image_name: compute_sim(image_name) for image_name in self.image_list}

        maxDet = p.maxDets[-1]
        self.evalImgs = [self.evaluate_image(image_name, areaRng, maxDet)
                         for areaRng in p.areaRng
                         for image_name in p.image_names
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def compute_sim(self, image_name):
        """Compute similarity for an image between ground truth and detected results
        """
        p = self.params
        gt = self._gts[image_name]
        dt = self._dts[image_name]

        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.simType == '3dpose':
            g = np.array([g['pose'] + [g['car_id']] for g in gt], dtype=np.float32)
            d = np.array([d['pose'] + [d['car_id']] for d in dt], dtype=np.float32)
        elif p.simType == '3dbbox':
            raise Exception('not implemented')
        else:
            raise Exception('unknown simType for similarity computation')

        # compute iou between each dt and gt region
        sims = euts.pose_similarity(d, g, p.shape_sim_mat)
        sims = np.stack(sims, axis=2)
        return sims

    def evaluate_image(self, image_name, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''

        def _satisfy(sim, cri):
            return sim[0] >= cri.shapeSim \
                   and sim[1] <= cri.transDis \
                   and sim[2] <= cri.oriDis

        p = self.params
        gt = self._gts[image_name]
        dt = self._dts[image_name]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]

        # load computed sims
        sims = self.sims[image_name][:, gtind, :] if len(self.sims[image_name]) > 0 else self.sims[image_name]

        # number of criterion
        T = len(p.shapeThrs)
        G = len(gt)
        D = len(dt)

        gtm = np.zeros((T, G))  # match of gt
        dtm = np.zeros((T, D))  # match of detection
        gtIg = np.array([g['_ignore'] for g in gt])  # ignore gt index
        dtIg = np.zeros((T, D))  # detection ignore

        # finding matches between detections & ground truth
        if not len(sims) == 0:
            for tind, cri in enumerate(p.criteria):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    cur_cri = cri
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, continue
                        if gtm[tind, gind] > 0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break

                        # continue to next gt unless better match made
                        if not _satisfy(sims[dind, gind], cur_cri):
                            continue
                        # if match successful and best so far, store appropriately
                        cur_match = sims[dind, gind]
                        cur_cri = Criterion(cur_match[0], cur_match[1], cur_match[2])

                        m = gind

                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue

                    dtIg[tind, dind] = gtIg[m]
                    gtm[tind, m] = d['id']
                    dtm[tind, dind] = gt[m]['id']

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

        # store results for given image and category
        return {
            'image_id': image_name,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        ''' Accumulate per image evaluation results and store the result in self.eval

        Inputs:
            p: input params for evaluation
        '''

        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.simThrs)  # number of thresh
        R = len(p.recThrs)  # number of recall thresh
        K = len(p.catIds) if p.useCats else 1  # number of categories
        A = len(p.areaRng)  # number of area scale
        M = len(p.maxDets)  # number of max detections per image
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(range(len(_pe.image_names)))  # can use to select image ids

        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(range(len(p.image_names))) if i in setI]
        A0 = len(_pe.areaRng)
        I0 = len(_pe.image_names)

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)

                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.evalRes = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, simThr=None, areaRng='all', maxDets=100):
            """Summarize an evaluation
            Input:
                ap: whether ask for average precision (ap = 1) or average recall.
                simThr: which sim criterion is considered.
                areaRng: which area range is considered.
                maxDets: max  number of detection per image
                f: the file handler for output stream

            """
            p = self.params
            iStr = ' {:<18} {} @[ Criteria={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.5f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            simstr = 'c0:c5' if simThr is None else simThr

            if not areaRng in p.areaRngLbl:
                print(iStr.format(titleStr, typeStr, simstr, areaRng, maxDets, -1.0))
                return -1.0

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.evalRes['precision']
                # sim matrix
                if simThr is not None:
                    t = [p.simThrs.index(simThr)]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.evalRes['recall']
                if simThr is not None:
                    t = p.simThrs.index(simThr)
                    s = s[t]
                s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, simstr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            out_names = ['AP', 'AP_c0', 'AP_c3', 'AP_s', 'AP_m', 'AP_l', 'AR_1',
                         'AR_10', 'AR_100', 'AR_s', 'AR_m', 'AR_l']

            stats = np.zeros(12)
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, simThr='c0', maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, simThr='c3', maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats, out_names

        if not self.evalRes:
            raise Exception('Please run accumulate() first')
        simType = self.params.simType
        if simType == '3dpose':
            summarize = _summarizeDets
        else:
            raise Exception('no givem simType %s' % simType)
        self.stats, metric_names = summarize()

        f = open(self.args.res_file, 'w')
        for name, value in zip(metric_names, self.stats):
            f.write('%s %.4f\n' % (name, value))
        f.close()

    def __str__(self):
        self.summarize()


class Params(object):
    """Params for apolloscape 3d car instance evaluation api
    Inputs:
        simType: currently only '3dpose' is supported. Later we may add '3dbbox' for eval
    """

    def __init__(self, simType='3dpose'):
        if simType == '3dpose':
            self.set_det_params()
        else:
            raise Exception('simType not supported')
        self.simType = simType

    def set_det_params(self):
        self.image_names = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.shapeThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.oriThrs = np.linspace(50, 5, np.round((50 - 5) / 5) + 1, endpoint=True)
        self.transThrs = np.linspace(2.8, 0.1, np.round((2.8 - 0.1) / .3) + 1, endpoint=True)
        self.shape_sim_mat = np.loadtxt('./utilities/sim_mat.txt')

        self.criterion_num = len(self.shapeThrs)
        self.simThrs = ['c' + str(i) for i in range(self.criterion_num)]

        assert self.criterion_num == len(self.oriThrs)
        assert self.criterion_num == len(self.transThrs)

        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        # from loss to strict criterion
        self.criteria = [Criterion(self.shapeThrs[i], self.transThrs[i],
                                   self.oriThrs[i]) for i in range(self.criterion_num)]

        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 64 ** 2], [64 ** 2, 192 ** 2], [192 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation self 3d car detection.')
    parser.add_argument('--test_dir', default='./test_eval_data/det3d_res/', help='the dir of results')
    parser.add_argument('--gt_dir', default='./test_eval_data/det3d_gt/', help='the dir of ground truth')
    parser.add_argument('--res_file', default='./test_eval_data/res.txt', help='the dir of ground truth')
    parser.add_argument('--simType', default=None, help='the type of evalution metric, default 3dpose')
    args = parser.parse_args()
    det_3d_metric = Detect3DEval(args)
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()
