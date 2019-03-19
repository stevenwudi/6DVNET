import torch
import os
from PIL import Image
import pickle
import numpy as np
# https://pypi.org/project/mat4py/
import mat4py
import numpy as np
import cv2

from maskrcnn_benchmark.structures.bounding_box import BoxList
from tqdm import tqdm


class Pascal3D(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_dir, list_flag, transforms, training):
        self.cfg = cfg.copy()
        self.dataset_dir = dataset_dir
        self.list_flag = list_flag
        self.transforms = transforms
        self.img_list_all = self.get_img_list()
        self.training = training

        # Load ground truth here
        print("Loading ground truth")
        target_file = os.path.join(self.dataset_dir, 'Annotations', 'car_imagenet.pth')
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
        self.car_CAD = mat4py.loadmat(os.path.join(self.dataset_dir, 'CAD', 'car.mat'))

        self.sub_label_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        # the following code is to investigate the sub_label
        # sub_label = set()
        # for t in self.targets:
        #     if type(t['record']['objects']['sub_label']) == list:
        #         sub_label.update(set(t['record']['objects']['sub_label']))
        #     else:
        #         sub_label.update(set([t['record']['objects']['sub_label']]))


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
        return {"height": 1, "width": 1}

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dataset_dir, 'Images', 'car_imagenet',  self.img_list_all[idx]+'.JPEG')).convert("RGB")
        image_shape = img.size

        #mat = mat4py.loadmat(os.path.join(self.dataset_dir, 'Annotations', 'car_imagenet', self.img_list_all[ idx ] + '.mat'))

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
        x2d = self.project_3d(x3d, target)

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

        save_set_dir = os.path.join(save_dir, settings)
        if not os.path.exists(save_set_dir):
            os.mkdir(save_set_dir)
        fig.savefig(os.path.join(save_dir, settings, image_name + '.png'), dpi=1)

    def project_3d(self, x3d, target):
        a = target['record']['objects']['viewpoint']['azimuth'] * np.pi / 180
        e = target['record']['objects']['viewpoint']['elevation'] * np.pi / 180
        d = target['record']['objects']['viewpoint']['distance']
        f = target['record']['objects']['viewpoint']['focal']
        theta = target['record']['objects']['viewpoint']['theta'] * np.pi / 180
        principle = [target['record']['objects']['viewpoint']['px'], target['record']['objects']['viewpoint']['py']]
        viewport = target['record']['objects']['viewpoint']['viewport']

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











