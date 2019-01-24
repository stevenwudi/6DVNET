from collections import namedtuple, OrderedDict
import numpy as np
import os
from PIL import Image
import json
import logging
import pickle
import torch
import cv2
from skimage import measure
from maskrcnn_benchmark.data.datasets import car_models
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.geometry import euler_angles_to_rotation_matrix, euler_angles_to_quaternions
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class Car3D(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, list_flag, transforms):
        """
        Constructor of ApolloScape helper class for reading and visualizing annotations.
        Modified from: https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/data.py
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.dataset_dir = dataset_dir
        self.list_flag = list_flag
        self.transforms = transforms
        self.img_list_all = []

        # Apollo 3d init
        Setting = namedtuple('Setting', ['image_name', 'data_dir'])
        setting = Setting([], self.dataset_dir)
        self.dataset = car_models.ApolloScape(setting)
        self._data_config = self.dataset.get_3d_car_config()
        self.car_id2name = car_models.car_id2name
        self.car_models = self.load_car_models()
        self.intrinsic_mat = self.get_intrinsic_mat()
        self.unique_car_models = np.array([2, 6, 7, 8, 9, 12, 14, 16, 18, 19, 20, 23, 25, 27, 28, 31, 32, 35, 37,
                                           40, 43, 46, 47, 48, 50, 51, 54, 56, 60, 61, 66, 70, 71, 76])
        self.unique_car_names = [self.car_id2name[x].name for x in self.unique_car_models]

    def get_img_list(self):
        """
        Get the image list,
        :param list_flag: ['train', 'val', test']
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :return:
        """
        if self.list_flag == "train":
            train_list_all = [line.rstrip('\n')[:-4] for line in open(os.path.join(self.dataset_dir, 'split', self.list_flag + '.txt'))]
            train_list_delete = [line.rstrip('\n') for line in open(os.path.join(self.dataset_dir, 'split', 'Mesh_overlay_train_error _delete.txt'))]
            print("Train delete %d images" % len(train_list_delete))

            self.img_list_all = [x for x in train_list_all if x not in train_list_delete]
        elif self.list_flag == "val":
            valid_list_all = [line.rstrip('\n')[:-4] for line in open(os.path.join(self.dataset_dir, 'split', 'val.txt'))]
            val_list_delete = [line.rstrip('\n') for line in open(os.path.join(self.dataset_dir, 'split', 'Mesh_overlay_val_error_delete.txt'))]
            self.img_list_all = [x for x in valid_list_all if x not in val_list_delete]
            print("Val delete %d images." % len(val_list_delete))

        elif self.list_flag == 'test':
            im_list = os.listdir(os.path.join(self.dataset_dir, 'images'))
            self.img_list_all = [x[:-4] for x in im_list]

        return self.img_list_all

    def load_car_models(self):
        """Load all the car models
        """
        car_models_all = OrderedDict([])
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            car_model = os.path.join(self.dataset_dir, 'car_models', model.name+'.pkl')
            # with open(car_model) as f:
            #     self.car_models[model.name] = pkl.load(f)
            #
            # This is a python 3 compatibility
            car_models_all[model.name] = pickle.load(open(car_model, "rb"), encoding='latin1')
            # fix the inconsistency between obj and pkl
            car_models_all[model.name]['vertices'][:, [0, 1]] *= -1
        return car_models_all

    def get_img_info(self, idx=None):
        # Di Wu temporally assume the fixed image size. It will be further examined.
        return {"height": 2710, "width": 3384}

    def get_intrinsic_mat(self):
        # intrinsic = self.dataset.get_intrinsic(image_name)
        # Intrinsic should always use camera 5
        intrinsic = self.dataset.get_intrinsic('Camera_5')
        intrinsic_mat = np.zeros((3, 3))
        intrinsic_mat[0, 0] = intrinsic[0]
        intrinsic_mat[1, 1] = intrinsic[1]
        intrinsic_mat[0, 2] = intrinsic[2]
        intrinsic_mat[1, 2] = intrinsic[3]
        intrinsic_mat[2, 2] = 1
        self.intrinsic_mat = intrinsic_mat
        return intrinsic_mat

    def __len__(self):
        return len(self.get_img_list())

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.dataset_dir, 'images', self.img_list_all[idx]+'.jpg')).convert("RGB")
        image_shape = img.size

        if self.list_flag in ['train', 'val']:
            target = self._add_gt_annotations_Car3d(idx, image_shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def _add_gt_annotations_Car3d(self, idx, image_shape):
        """Add ground truth annotation metadata to an roidb entry."""
        # initiate the lists
        masks = []
        boxes = []
        car_cat_classes = []
        poses = []
        quaternions = []

        car_pose_file = os.path.join(self.dataset_dir, 'car_poses', self.img_list_all[idx] + '.json')
        assert os.path.exists(car_pose_file), 'Label \'{}\' not found'.format(car_pose_file)
        with open(car_pose_file) as f:
            car_poses = json.load(f)

        for i, car_pose in enumerate(car_poses):
            car_name = self.car_id2name[car_pose['car_id']].name
            car = self.car_models[car_name]
            pose = np.array(car_pose['pose'])

            # project 3D points to 2d image plane
            rot_mat = euler_angles_to_rotation_matrix(pose[:3])
            rvect, _ = cv2.Rodrigues(rot_mat)
            imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], self.intrinsic_mat, distCoeffs=None)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # project 3D points to 2d image plane
            mask = np.zeros((image_shape[1], image_shape[0]))
            for face in car['faces'] - 1:
                pts = np.array([[imgpts[idx, 0], imgpts[idx, 1]] for idx in face], np.int32)
                pts = pts.reshape((-1, 1, 2))
                #cv2.polylines(mask, [pts], True, (255, 255, 255))
                cv2.drawContours(mask, [pts], 0, (255, 255, 255), -1)

            # Find mask
            contours = measure.find_contours(np.array(mask), 0.5)
            mask_instance = []
            # if len(contours) > 1:
            #     print("Image with problem: %d, %s" % (idx, self.img_list_all[idx]))
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                mask_instance.append(segmentation)

            masks.append(mask_instance)
            #cv2.imwrite(os.path.join('/media/SSD_1TB/ApolloScape', self.img_list_all[idx] + '_' + str(i) + '_.jpg'), mask)
            x1, y1, x2, y2 = imgpts[:, 0].min(), imgpts[:, 1].min(), imgpts[:, 0].max(), imgpts[:, 1].max()
            boxes.append([x1, y1, x2, y2])
            quaternions.append(euler_angles_to_quaternions(np.array(car_pose['pose'][:3]))[0])
            car_cat_classes.append(np.where(self.unique_car_models == car_pose['car_id'])[0][0])
            poses.append(car_pose['pose'])

        target = BoxList(boxes, image_shape, mode="xyxy")
        masks = SegmentationMask(masks, image_shape)

        car_cat_classes = torch.tensor(car_cat_classes)
        target.add_field('car_cat_classes', car_cat_classes)
        target.add_field("masks", masks)
        quaternions = torch.tensor(quaternions)
        target.add_field("quaternions", quaternions)
        poses = torch.tensor(poses)
        target.add_field("poses", poses)

        labels = np.ones(car_cat_classes.shape)
        labels = torch.tensor(labels)
        target.add_field("labels", labels)

        target = target.clip_to_image(remove_empty=True)

        return target

