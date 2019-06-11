from maskrcnn_benchmark.config import cfg
from predictor import KittiDemo
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

config_file = "../configs/e2e_3d_car_101_FPN_triple_head.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "/media/SSD_1TB/ApolloScape/6DVNET_experiments/e2e_3d_car_101_FPN_triple_head/May31-05-30_n606_step/model_final.pth"])

coco_demo = KittiDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
)

category_id_to_contiguous_id = {'background': 0, 'car': 1}

coco_demo.CATEGORIES = list(category_id_to_contiguous_id.keys())

# load image and then run prediction
img_dir = "/media/SSD_1TB/ApolloScape/ECCV2018_apollo/test/images"
img_files = sorted(os.listdir(img_dir))
img = Image.open(os.path.join(img_dir, img_files[200])).convert("RGB")
image = np.array(img)[:, :, [2, 1, 0]]

predictions = coco_demo.run_on_opencv_image(image)

imshow(predictions)