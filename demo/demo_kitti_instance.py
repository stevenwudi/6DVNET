from maskrcnn_benchmark.config import cfg
from predictor import KittiDemo
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

config_file = "../configs/e2e_mask_rcnn_R_101_FPN_1x_kitti_instance.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "/media/SSD_1TB/Kitti/experiments/e2e_mask_rcnn_R_101_FPN_1x_kitti_instance/Jan10-23-02_n606_step/model_0002500.pth"])

coco_demo = KittiDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
)

category_id_to_contiguous_id = {
              'background': 0,
              'bicycle': 1,
              'bus': 2,
              'car': 3,
              'caravan': 4,
              'motorcycle': 5,
              'person': 6,
              'rider': 7,
              'trailer': 8,
              'train': 9,
              'truck': 10}

coco_demo.CATEGORIES = list(category_id_to_contiguous_id.keys())

# load image and then run prediction
img_dir = "/media/SSD_1TB/Kitti/data_semantics/testing/image_2"
img_files = os.listdir(img_dir)
img = Image.open(os.path.join(img_dir, img_files[1])).convert("RGB")
image = np.array(img)[:, :, [2, 1, 0]]

predictions = coco_demo.run_on_opencv_image(image)

imshow(predictions)