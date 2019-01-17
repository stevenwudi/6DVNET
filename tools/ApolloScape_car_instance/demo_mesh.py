"""

"""
import os
from tqdm import tqdm
from collections import namedtuple
from tools.ApolloScape_car_instance.render_car_mesh import CarPoseVisualizer

Setting = namedtuple('Setting', ['image_name', 'data_dir'])

set_name = 'train'   #['train', 'val']
# You need to specify the dataset dir
dataset_dir = '/media/SSD_1TB/ApolloScape/ECCV2018_apollo/train/'

img_list = [line.rstrip('\n')[:-4] for line in open(os.path.join(dataset_dir, 'split', set_name + '.txt'))]
save_dir = os.path.join(dataset_dir, 'Mesh_overlay')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

setting = Setting(None, dataset_dir)
visualizer = CarPoseVisualizer(setting)
visualizer.load_car_models()

img_list = ['171206_034636094_Camera_5']
#img_list = ['180114_024339575_Camera_5']
car_pose_dir = dataset_dir + 'car_poses/'
car_pose_file = car_pose_dir + '%s.json' % img_list[0]

for img in tqdm(img_list):
    setting = Setting(img, dataset_dir)
    visualizer.set_dataset(setting)
    merged_image = visualizer.showAnn_image(setting.image_name, car_pose_file, set_name, save_dir)
