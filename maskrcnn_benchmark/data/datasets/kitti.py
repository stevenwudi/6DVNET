import os
from PIL import Image
import numpy as np
import torch
from maskrcnn_benchmark.data.datasets.devkit_semantics.devkit.helpers.labels import labels, id2label
from maskrcnn_benchmark.config import cfg
import pycocotools
from skimage import measure
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.bounding_box import BoxList


class KittiInstanceDataset(torch.utils.data.Dataset):
    """
    The labels IDs, names and instance classes of the Cityscapes dataset are
    used and can be found [here]
    (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)

    """

    def __init__(self, root, ann_file, img_dir, transforms=None):
        self.root = root
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms

        self.img_files = os.listdir(self.img_dir)
        self.anno_files = os.listdir(self.ann_file)

        self.category_id_to_contiguous_id = {'bicycle': 1,
                                             'bus': 2,
                                             'car': 3,
                                             'caravan': 4,
                                             'motorcycle': 5,
                                             'person': 6,
                                             'rider': 7,
                                             'trailer': 8,
                                             'train': 9,
                                             'truck': 10}

        if ann_file and len(self.img_files) != len(self.anno_files):
            assert "Annotation has %d files and Images has %d files." % len(self.anno_files), len(self.img_files)

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.img_dir, self.img_files[idx])).convert("RGB")
        image_shape = img.size
        lable_img = Image.open(os.path.join(self.ann_file, self.anno_files[idx]))
        lable_img = np.array(lable_img)

        target = self.target_png_to_boxlist(lable_img, image_shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.anno_files)

    def get_img_info(self, index=None):
        # Di Wu temporally assume the fixed image size. It will be further examined.
        return {"height": 375, "width": 1242}

    def target_png_to_boxlist(self, l_img, image_shape):
        # initiate the lists
        masks = []
        boxes = []
        classes = []
        for label in np.unique(l_img):
            class_id = label // 256
            if id2label[class_id].hasInstances:
                area = np.sum(l_img == label)
                if area < cfg.TRAIN.GT_MIN_AREA:
                    continue
                # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
                mask = l_img == label
                # mask_f = np.array(mask, order='F', dtype=np.uint8)
                # rle = pycocotools.mask.encode(mask_f)
                # ground_truth_bounding_box = pycocotools.mask.toBbox(rle)

                contours = measure.find_contours(np.array(mask), 0.5)
                mask_instance = []
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    mask_instance.append(segmentation)

                xd, yd = np.where(mask)
                x1, y1, x2, y2 = yd.min(), xd.min(), yd.max(), xd.max()

                boxes.append([x1, y1, x2, y2])
                masks.append(mask_instance)
                classes.append(self.category_id_to_contiguous_id[id2label[class_id].name])

        target = BoxList(boxes, image_shape, mode="xyxy")
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        masks = SegmentationMask(masks, image_shape)
        target.add_field("masks", masks)
        target = target.clip_to_image(remove_empty=True)

        return target