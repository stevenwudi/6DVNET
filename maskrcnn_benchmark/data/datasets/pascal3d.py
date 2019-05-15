import torch
import os
from PIL import Image
import pickle
# https://pypi.org/project/mat4py/
import mat4py
import numpy as np
import cv2
from skimage import measure
from matplotlib import pyplot as plt
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from tqdm import tqdm
from pycocotools import mask as maskUtils
import copy
import time
from collections import defaultdict
import itertools
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Pascal3D(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_dir, list_flag, transforms, training, load_cad=True):
        self.cfg = cfg.copy()
        self.dataset_dir = dataset_dir
        self.list_flag = list_flag
        self.transforms = transforms
        self.img_list_all = self.get_img_list()
        self.training = training

        # Load ground truth here
        target_file = os.path.join(self.dataset_dir, 'Annotations', 'car_imagenet_' + self.list_flag +'.pth')
        if os.path.isfile(target_file):
            with open(target_file, 'rb') as f:
                targets = pickle.load(f)
        else:
            targets = []
            for im_name in tqdm(self.img_list_all):
                mat = mat4py.loadmat(os.path.join(self.dataset_dir, 'Annotations', 'car_imagenet', im_name + '.mat'))
                targets.append(mat)
            with open(target_file, 'wb') as f:
                pickle.dump(targets, f)
        self.targets = targets

        # load car CAD model
        if load_cad:
            print("Loading ground CAD")
            self.car_CAD = mat4py.loadmat(os.path.join(self.dataset_dir, 'CAD', 'car.mat'))

        self.sub_label_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.category_to_id_map = {'car': 1}
        self.eval_class = [1]
        # the following code is to investigate the sub_label
        # sub_label = set()
        # for t in self.targets:
        #     if type(t['record']['objects']['sub_label']) == list:
        #         sub_label.update(set(t['record']['objects']['sub_label']))
        #     else:
        #         sub_label.update(set([t['record']['objects']['sub_label']]))

        # The following code is to investigate the image size
        # width_min, width_max, height_min, height_max = (160, 3072, 110, 2048)
        # width_min, width_max, height_min, height_max = 1e5, 0, 1e5, 0
        # for t in self.targets:
        #     height = t['record']['size']['height']
        #     width = t['record']['size']['width']
        #     width_min = min(width, width_min)
        #     width_max = max(width, width_max)
        #     height_min = min(height, height_min)
        #     height_max = max(height, height_max)

    def get_img_list(self):
        """
        Get the image list,
        :param list_flag: ['train', 'val', test']
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :return:
        """
        if self.list_flag == "train":
            self.img_list_all = [line.rstrip('\n') for line in open(os.path.join(self.dataset_dir, 'Image_sets/car_imagenet_' + self.list_flag + '.txt'))]
            print("Number of Train image: %d." % len(self.img_list_all))

        elif self.list_flag == "val":
            self.img_list_all = [line.rstrip('\n') for line in open(os.path.join(self.dataset_dir, 'Image_sets/car_imagenet_' + self.list_flag + '.txt'))]
            print("Number of val image: %d." % len(self.img_list_all))

        return self.img_list_all

    def __len__(self):
        return len(self.get_img_list())

    def get_img_info(self, idx=None):
        # Di Wu temporally assume the fixed image size. It will be further examined.
        return {"height": self.targets[idx]['record']['size']['height'],
                "width":  self.targets[idx]['record']['size']['width']}

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dataset_dir, 'Images', 'car_imagenet',  self.img_list_all[idx]+'.JPEG')).convert("RGB")
        image_shape = img.size

        target = self._add_gt_annotations_Pascal3D(idx)

        # We also change the size of image very iteration:
        if self.training:
            resize_ratio = np.random.uniform(low=self.cfg['INPUT']['MIN_SIZE_TRAIN_RATIO'][0],
                                             high=self.cfg['INPUT']['MIN_SIZE_TRAIN_RATIO'][1])
            self.transforms.transforms[0].min_size = int(min(image_shape) * resize_ratio)
        else:
            self.transforms.transforms[0].min_size = min(image_shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def _add_gt_annotations_Pascal3D(self, idx):
        """Add ground truth annotation metadata to an roidb entry."""
        # initiate the lists
        boxes = []
        cad_classes = []
        masks = []
        segms = []  # This is only for test evaluation
        azimuths = []
        elevations = []
        distances = []
        rot_vects = []

        target = self.targets[idx]

        image_shape = (target['record']['size']['width'], target['record']['size']['height'])
        if type(target['record']['objects']['cad_index']) == list:
            # there are multiple objects
            for index in range(len(target['record']['objects']['cad_index'])):
                # Originally it is matlab implemenation, index starts from 1...
                cad_index = target['record']['objects']['cad_index'][index] - 1
                bbox = target['record']['objects']['bbox'][index]
                boxes.append(bbox)
                cad_classes.append(cad_index)

                if self.cfg['MODEL']['CAR_CLS_HEAD_ON']:
                    azimuth = target['record']['objects']['viewpoint'][index]['azimuth']
                    elevation = target['record']['objects']['viewpoint'][index]['elevation']
                    distance = target['record']['objects']['viewpoint'][index]['distance']
                    azimuths.append(azimuth/360.)
                    elevations.append(elevation/360.)
                    rot_vects.append([azimuth/360., elevation/360.])
                    distances.append(distance)

                if self.cfg['MODEL']['MASK_ON']:
                    # Originally it is matlab implemenation, index starts from 1...
                    mask_instance, encoded_ground_truth = self.extract_mask_and_segms(target, image_shape, index)
                    masks.append(mask_instance)
                    segms.append(encoded_ground_truth)

            target_boxlist = BoxList(boxes, image_shape, mode="xyxy")
            cad_classes = torch.tensor(cad_classes)
            target_boxlist.add_field('cad_classes', cad_classes)
            labels = np.ones(cad_classes.shape)
            labels = torch.tensor(labels)
            target_boxlist.add_field("labels", labels)
            if self.cfg['MODEL']['MASK_ON']:
                masks = SegmentationMask(masks, image_shape)
                target_boxlist.add_field("masks", masks)
                target_boxlist = target_boxlist.clip_to_image(remove_empty=True)
                if not self.training:
                    target_boxlist.add_field("segms", segms)
            if self.cfg['MODEL']['CAR_CLS_HEAD_ON']:
                target_boxlist.add_field("azimuth", torch.tensor(azimuths))
                target_boxlist.add_field("elevation", torch.tensor(elevations))
                target_boxlist.add_field("rot_vects", torch.tensor(rot_vects))

                target_boxlist.add_field("distance", torch.tensor(distances))

        else:
            # Originally it is matlab implemenation, index starts from 1...
            cad_index = target['record']['objects']['cad_index'] - 1
            bbox = target['record']['objects']['bbox']
            boxes.append(bbox)
            target_boxlist = BoxList(boxes, image_shape, mode="xyxy")

            cad_classes.append(cad_index)
            cad_classes = torch.tensor(cad_classes)
            target_boxlist.add_field('cad_classes', cad_classes)

            labels = np.ones(cad_classes.shape)
            labels = torch.tensor(labels)
            target_boxlist.add_field("labels", labels)
            if self.cfg['MODEL']['MASK_ON']:
                # Originally it is matlab implemenation, index starts from 1...
                mask_instance, encoded_ground_truth = self.extract_mask_and_segms(target, image_shape)
                masks = SegmentationMask([mask_instance], image_shape)
                target_boxlist.add_field("masks", masks)
                target_boxlist = target_boxlist.clip_to_image(remove_empty=True)
                if not self.training:
                    target_boxlist.add_field("segms", [encoded_ground_truth])
            if self.cfg['MODEL']['CAR_CLS_HEAD_ON']:
                azimuth = target['record']['objects']['viewpoint']['azimuth']
                elevation = target['record']['objects']['viewpoint']['elevation']
                azimuths.append(azimuth/360.)
                elevations.append(elevation/360.)
                rot_vects.append([azimuth/360., elevation / 360.])

                distances.append(target['record']['objects']['viewpoint']['distance'])

                target_boxlist.add_field("azimuth", torch.tensor(azimuths))
                target_boxlist.add_field("elevation", torch.tensor(elevations))
                target_boxlist.add_field("rot_vects", torch.tensor(rot_vects))

                target_boxlist.add_field("distance", torch.tensor(distances))

        labels = np.ones(cad_classes.shape)
        labels = torch.tensor(labels)
        target_boxlist.add_field("labels", labels)

        return target_boxlist

    def extract_mask_and_segms(self, target, image_shape, index=None):
        if index is not None:
            cad_index = target['record']['objects']['cad_index'][index] - 1
        else:
            cad_index = target['record']['objects']['cad_index'] - 1

        vertices = self.car_CAD['car']['vertices'][cad_index]
        faces = np.array(self.car_CAD['car']['faces'][cad_index])

        x3d = np.array(vertices)
        x2d = self.project_3d(x3d, target, index)
        mask = np.zeros(image_shape[::-1])
        for face in faces - 1:
            pts = np.array([[x2d[f, 0], x2d[f, 1]] for f in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            #cv2.polylines(mask, [pts], True, (0, 255, 0))
            cv2.drawContours(mask, [pts], 0, (255, 255, 255), -1)

        # # fill some holes here
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours = measure.find_contours(np.array(mask), 0.5)
        mask_instance = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            mask_instance.append(segmentation)

        # Find mask
        if not self.training:
            ground_truth_binary_mask = np.zeros(mask.shape, dtype=np.uint8)
            ground_truth_binary_mask[mask == 255] = 1
            fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
            encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
        else:
            encoded_ground_truth = []

        return mask_instance, encoded_ground_truth

    def show_car_overlay(self, idx):
        # Show CAD overlay with and image, modify from the original .m file
        # load image
        #img = Image.open(os.path.join(self.dataset_dir, 'Images', 'car_imagenet',  self.img_list_all[idx]+'.JPEG')).convert("RGB")

        img = cv2.imread(os.path.join(self.dataset_dir, 'Images', 'car_imagenet',  self.img_list_all[idx]+'.JPEG'), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        target = self.targets[idx]

        # load CAD model
        if target['record']['objects']['viewpoint']['distance'] == 0:
            print('No continuous viewpoint')
            return

        # Originally it is matlab implemenation, index starts from 1...
        cad_index = target['record']['objects']['cad_index'] - 1
        vertices = self.car_CAD['car']['vertices'][cad_index]
        faces = np.array(self.car_CAD['car']['faces'][cad_index])

        x3d = np.array(vertices)
        x2d = self.project_3d(x3d, target, idx)

        mask = np.zeros(img.shape)
        for face in faces-1:
            pts = np.array([[x2d[idx, 0], x2d[idx, 1]] for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], True, (0, 255, 0))

        merged_image = img.copy()
        alpha = 0.8
        cv2.addWeighted(img.astype(np.uint8), 1.0, mask.astype(np.uint8), alpha, 0, merged_image)

        from matplotlib import pyplot as plt
        # Save figure
        plt.close('all')
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(image.shape[1]/10, image.shape[0]/10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(merged_image)
        plt.show()
        # save_set_dir = os.path.join(save_dir, settings)
        # if not os.path.exists(save_set_dir):
        #     os.mkdir(save_set_dir)
        # fig.savefig(os.path.join(save_dir, settings, image_name + '.png'), dpi=1)

    def project_3d(self, x3d, target, index):
        if index is not None:
            viewpoint = target['record']['objects']['viewpoint'][index]
        else:
            viewpoint = target['record']['objects']['viewpoint']

        a = viewpoint['azimuth'] * np.pi / 180
        e = viewpoint['elevation'] * np.pi / 180
        d = viewpoint['distance']
        f = viewpoint['focal']
        theta = viewpoint['theta'] * np.pi / 180
        principle = [viewpoint['px'], viewpoint['py']]
        viewport = viewpoint['viewport']

        # camera centre
        C = np.zeros((3, 1))
        C[0] = d * np.cos(e) * np.sin(a)
        C[1] = -d * np.cos(e) * np.cos(a)
        C[2] = d * np.sin(e)

        # Rotate coordinate system by thea is equal to rotating the model by -theta
        a = -a
        e = - (np.pi/2 - e)

        # Rotation matrix
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])  # rotate by a
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])  # rotate by e
        R = np.matmul(Rx, Rz)

        # Perspective project matrix
        # However, we set the viewport to 3000, which makes the camera similar to
        # an affine-camera. Exploring a real perspective camera can be a future work
        M = viewport
        P = np.matmul(([[M*f, 0, 0], [0, M*f, 0], [0, 0, -1]]),  np.hstack((R, np.matmul(-R, C))))
        # project
        x = np.matmul(P, np.hstack((x3d, np.ones((len(x3d), 1)))).transpose(1, 0))

        x[0, :] /= x[2, :]
        x[1, :] /= x[2, :]
        x = x[0:2, :]

        # Rotation matrix in 2D
        R2d = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x = np.matmul(R2d, x).transpose(1, 0)

        # transform to image coordinates
        x[:, 1] *= -1
        x = x + np.tile(np.array(principle), [x.shape[0], 1])

        return x

    def loadGt(self):
        """
        Load result file and return a result api object.
        :param: type      : boxes, or segms
        """
        print('Loading and preparing results...')
        res = Pascal3D(self.cfg, self.dataset_dir, self.list_flag, self.transforms, self.training, load_cad=False)
        res.dataset = dict()
        res.dataset['categories'] = copy.deepcopy(self.category_to_id_map)
        res.dataset['images'] = []
        anns = []
        count = 1
        tic = time.time()
        for idx in tqdm(range(len(self.img_list_all))):
            res.dataset['images'].append({'id': self.img_list_all[idx]})
            if self.list_flag in ['train', 'val']:
                target = self._add_gt_annotations_Pascal3D(idx)

                for id in range(len(target)):
                    ann = dict()
                    ann['image_id'] = self.img_list_all[idx]
                    ann['category_id'] = int(target.get_field('labels')[id].numpy())
                    bb = target.bbox[id].numpy()
                    x1, x2, y1, y2 = bb[0], bb[2], bb[1], bb[3]
                    w = x2 - x1
                    h = y2 - y1
                    x_c = (x1 + x2)/2
                    y_c = (y1 + y2)/2
                    ann['bbox'] = [x_c, y_c, w, h]
                    ann['area'] = w * h

                    if target.has_field('segms'):
                        ann['segms'] = target.get_field('segms')[id]
                        # now only support compressed RLE format as segmentation results
                        ann['area'] = maskUtils.area(ann['segms'])
                        if not 'boxes' in ann:
                            ann['boxes'] = maskUtils.toBbox(ann['segms'])

                    ann['id'] = count
                    ann['iscrowd'] = 0
                    count += 1
                    anns.append(ann)

        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def loadRes(self, predictions):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        print('Loading and preparing results...')
        res = Pascal3D(self.cfg, self.dataset_dir, self.list_flag, self.transforms, self.training, load_cad=False)
        res.dataset = dict()
        res.dataset['categories'] = copy.deepcopy(self.category_to_id_map)
        res.dataset['images'] = []
        anns = []
        count = 1
        tic = time.time()
        masker = Masker(threshold=0.5, padding=1)

        for idx in tqdm(range(len(self.img_list_all))):
            res.dataset['images'].append({'id': self.img_list_all[idx]})
            prediction = predictions[idx]
            for id in range(len(prediction)):
                ann = dict()
                ann['image_id'] = self.img_list_all[idx]
                ann['category_id'] = int(prediction.get_field('labels')[id].cpu().numpy())
                bb = prediction.bbox[id].cpu().numpy()
                x1, x2, y1, y2 = bb[0], bb[2], bb[1], bb[3]
                w = x2 - x1
                h = y2 - y1
                x_c = (x1 + x2) / 2
                y_c = (y1 + y2) / 2
                ann['bbox'] = [x_c, y_c, w, h]
                ann['area'] = w * h
                ann['score'] = prediction.get_field('scores')[id].cpu().numpy()

                if prediction.has_field('mask'):
                    image_width, image_height, _ = res.targets[idx]['record']['imgsize']

                    masks = prediction.get_field('mask')
                    # Masker is necessary only if masks haven't been already resized.
                    if list(masks.shape[-2:]) != [image_height, image_width]:
                        masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
                        masks = masks[0]

                    ann['segms'] = [maskUtils.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0] for mask in masks]
                    ann['category_id'] = int(prediction.get_field('labels')[id].cpu().numpy())
                    # now only support compressed RLE format as segmentation results
                    #ann['area'] = maskUtils.area(ann['segms'])
                    if not 'boxes' in ann:
                        ann['boxes'] = maskUtils.toBbox(ann['segms'])

                ann['id'] = count
                count += 1
                ann['iscrowd'] = 0
                anns.append(ann)

        print('DONE (t={:0.2f}s)'.format(time.time() - tic))
        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        # if 'categories' in self.dataset:
        #     for cat in self.dataset['categories']:
        #         cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids
