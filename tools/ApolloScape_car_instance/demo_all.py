import os
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from tools.ApolloScape_car_instance.render_car_mesh import CarPoseVisualizer

Setting = namedtuple('Setting', ['image_name', 'data_dir'])

set_name = 'train'   #['train', 'val']
# You need to specify the dataset dir
dataset_dir = '/media/SSD_1TB/ApolloScape/ECCV2018_apollo/train/'

img_list = [line.rstrip('\n')[:-4] for line in open(os.path.join(dataset_dir, 'split', set_name + '.txt'))]
save_dir = os.path.join(dataset_dir, 'Mesh_overlay')

setting = Setting(None, dataset_dir)
visualizer = CarPoseVisualizer(setting)
visualizer.load_car_models()


### Find car models ########
if False:
    car_models = []
    for img in tqdm(img_list):
        setting = Setting(img, dataset_dir)
        visualizer.set_dataset(setting)
        car_id = visualizer.findCarModels(setting.image_name)
        car_models.append(car_id)

    car_models = np.array(np.hstack(car_models))
    print("Car model: max: %d, min: %d, total: %d" % (car_models.max(), car_models.min(), len(car_models)))
    # Car model: max: 76, min: 2, total: 50445
    np.unique(car_models)
    # array([ 2,  6,  7,  8,  9, 12, 14, 16, 18, 19, 20, 23, 25, 27, 28, 31, 32,
    #        35, 37, 40, 43, 46, 47, 48, 50, 51, 54, 56, 60, 61, 66, 70, 71, 76])
    len(np.unique(car_models))
    # 34

    # we calculate the car frequencies as below:
    total_count = np.sum([visualizer.car_counts[car_id]['car_counts'] for car_id in visualizer.car_counts.keys()])
    for car_id in sorted(visualizer.car_counts.keys()):
        visualizer.car_counts[car_id]['inv_freq'] = total_count / visualizer.car_counts[car_id]['car_counts']

    np.round(np.array([visualizer.car_counts[car_id]['inv_freq'] / 10 for car_id in sorted(visualizer.car_counts.keys())]),2)
    # [ 1.2 ,  3.74, 12.03,  5.08,  6.69, 11.7 , 20.25,  0.37, 14.32,
    #    14.24, 84.02, 14.78,  9.46, 13.62,  1.51,  9.96, 43.84,  1.83,
    #     4.65,  6.62, 11.54,  0.6 ,  6.73, 29.48,  7.82,  5.55, 12.36,
    #    25.85,  8.63,  6.49,  4.13,  2.6 ,  6.3 , 10.84]

### Collect Pose statistics ###
if False:
    poses = []
    for img in tqdm(img_list):
        setting = Setting(img, dataset_dir)
        visualizer.set_dataset(setting)
        pose = visualizer.collect_pose(setting.image_name)
        poses.append(pose)

    poses_array = [np.vstack(p) for p in poses]
    poses_array = np.array(np.vstack(poses_array))
    for i in range(poses_array.shape[1]):
        print(poses_array[:, i].min(), poses_array[:, i].max(), poses_array[:, i].mean(), poses_array[:, i].std())

    """
       min        max        mean       std
    -3.14052    3.14085     1.1784      1.3378
    -1.55991    1.55723     0.0298      0.4886      
    -3.14159    3.14158     -1.323      2.0161
    -145.807    518.745     -3.756      15.005
    -0.982025   689.119     9.9432      7.0902
    1.5389      3493.44     54.044      41.8559
    -0.09963    0.975609    0.0458      0.03081
    -0.99988    0.999956    0.4072      0.45224
    -0.84664    0.819640    -0.0282     0.05850
    -0.99997    0.999731    -0.4341     0.65865
    """

### Find the area ######
if False:
    areas = []
    for img in tqdm(img_list):
        setting = Setting(img, dataset_dir)
        visualizer.set_dataset(setting)
        area = visualizer.findArea(setting.image_name)
        areas.append(area)

    areas = np.array(np.hstack(areas))
    print("Area: max: %d, min: %d, mean: %d, total: %d" % (areas.max(), areas.min(), areas.mean(), len(areas)))
    # Area: max: 2271480, min: 1, mean: 28393, total: 50445

############## Visualisation ####################
if True:
    # We extract only Camera6 here
    # img_list = [line.rstrip('\n')[:-4] for line in open(os.path.join(dataset_dir, 'split', set_name + '.txt')) if
    #             line.rstrip('\n')[:-4][-1] == '6']
    # save_dir = os.path.join(dataset_dir, 'Mesh_overlay')
    img_list = ['171206_034636094_Camera_5']
    for img in tqdm(img_list):
        setting = Setting(img, dataset_dir)
        visualizer.set_dataset(setting)
        merged_image = visualizer.showAnn(setting.image_name, set_name, save_dir)

### Translation finding via 2d to 3D
if True:
    dis_trans_all = []
    for img in tqdm(img_list):
        setting = Setting(img, dataset_dir)
        visualizer.set_dataset(setting)
        dis_trans = visualizer.findTrans(setting.image_name)
        dis_trans_all.append(np.array(dis_trans))
        print(np.array(dis_trans).mean())

print(np.hstack(dis_trans_all).mean())
# 6.20518713885198