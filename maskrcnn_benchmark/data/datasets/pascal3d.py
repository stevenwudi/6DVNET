import torch
import os
from PIL import Image
# https://pypi.org/project/mat4py/
import mat4py

from maskrcnn_benchmark.structures.bounding_box import BoxList
from tqdm import tqdm


class Pascal3D(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_dir, list_flag, transforms, training):
        self.cfg = cfg.copy()
        self.dataset_dir = dataset_dir
        self.list_flag = list_flag
        self.transforms = transforms
        self.img_list_all = []
        self.training = training

        # Load ground truth here
        print("Loading ground truth")
        targets = []
        for im_name in tqdm(self.img_list_all):
            mat = mat4py.loadmat(os.path.join(self.dataset_dir, 'Annotations', 'car_imagenet', im_name + '.mat'))
            targets.append(mat)
        self.targets = targets

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





