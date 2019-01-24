"""
    Brief: Demo for render labelled car 3d poses to the image
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import argparse
import cv2
from tools.ApolloScape_car_instance import car_models
from tools.ApolloScape_car_instance import data
import numpy as np
import json
import pickle as pkl

import tools.ApolloScape_car_instance.utils.utils as uts
import tools.ApolloScape_car_instance.utils.eval_utils as eval_uts
import logging
from collections import OrderedDict

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CarPoseVisualizer(object):
    def __init__(self, args=None, scale=1.0, linewidth=0.):
        """Initializer
        Input:
            scale: whether resize the image in case image is too large
            linewidth: 0 indicates a binary mask, while > 0 indicates
                       using a frame.
        """
        self.dataset = data.ApolloScape(args)
        self._data_config = self.dataset.get_3d_car_config()

        self.MAX_DEPTH = 1e4
        self.MAX_INST_NUM = 100
        h, w = self._data_config['image_size']

        # must round prop to 4 due to renderer requirements
        # this will change the original size a bit, we usually need rescale
        # due to large image size
        self.image_size = np.uint32(uts.round_prop_to(np.float32([h * scale, w * scale])))
        self.scale = scale
        self.linewidth = linewidth
        self.colors = np.random.random((self.MAX_INST_NUM, 3)) * 255

        self.car_counts = {}

    def set_dataset(self, args):
        self.dataset = data.ApolloScape(args)
        self._data_config = self.dataset.get_3d_car_config()

    def load_car_models(self):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            car_model = '%s%s.pkl' % (self._data_config['car_model_dir'], model.name)
            # with open(car_model) as f:
            #     self.car_models[model.name] = pkl.load(f)
            #
            # This is a python 3 compatibility
            self.car_models[model.name] = pkl.load(open(car_model, "rb"), encoding='latin1')
            # fix the inconsistency between obj and pkl
            self.car_models[model.name]['vertices'][:, [0, 1]] *= -1
            # print("Vertice number is: %d" % len(self.car_models[model.name]['vertices']))
            # print("Face number is: %d" % len(self.car_models[model.name]['faces']))

    def render_car_cv2(self, pose, car_name, image):
        """Render a car instance given pose and car_name
        """
        car = self.car_models[car_name]
        pose = np.array(pose)
        # project 3D points to 2d image plane
        rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
        rvect, _ = cv2.Rodrigues(rmat)
        imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], self.intrinsic, distCoeffs=None)

        mask = np.zeros(image.shape)
        for face in car['faces'] - 1:
            pts = np.array([[imgpts[idx, 0, 0], imgpts[idx, 0, 1]] for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], True, (0, 255, 0))

        ### Open3d
        if False:
            from open3d import TriangleMesh, Vector3dVector, Vector3iVector, draw_geometries
            mesh = TriangleMesh()
            mesh.vertices = Vector3dVector(car['vertices'])
            mesh.triangles = Vector3iVector(car['faces'])
            mesh.paint_uniform_color([1, 0.706, 0])

            car_v2 = np.copy(car['vertices'])
            car_v2[:, 2] += car_v2[:, 2].max()*3
            mesh2 = TriangleMesh()
            mesh2.vertices = Vector3dVector(car_v2)
            mesh2.triangles = Vector3iVector(car['faces'])
            mesh2.paint_uniform_color([0, 0.706, 0])
            draw_geometries([mesh, mesh2])

            print("A mesh with no normals and no colors does not seem good.")

            print("Painting the mesh")
            mesh.paint_uniform_color([1, 0.706, 0])
            draw_geometries([mesh])


            print("Computing normal and rendering it.")
            mesh.compute_vertex_normals()
            print(np.asarray(mesh.triangle_normals))
            draw_geometries([mesh])

        return mask

    def compute_reproj_sim(self, car_names, out_file=None):
        """Compute the similarity matrix between every pair of cars.
        """
        if out_file is None:
            out_file = './sim_mat.txt'

        sim_mat = np.eye(len(self.car_model))
        for i in range(len(car_names)):
            for j in range(i, len(car_names)):
                name1 = car_names[i][0]
                name2 = car_names[j][0]
                ind_i = self.car_model.keys().index(name1)
                ind_j = self.car_model.keys().index(name2)
                sim_mat[ind_i, ind_j] = self.compute_reproj(name1, name2)
                sim_mat[ind_j, ind_i] = sim_mat[ind_i, ind_j]

        np.savetxt(out_file, sim_mat, fmt='%1.6f')

    def compute_reproj(self, car_name1, car_name2):
        """Compute reprojection error between two cars
        """
        sims = np.zeros(10)
        for i, rot in enumerate(np.linspace(0, np.pi, num=10)):
            pose = np.array([0, rot, 0, 0, 0, 5.5])
            depth1, mask1 = self.render_car(pose, car_name1)
            depth2, mask2 = self.render_car(pose, car_name2)
            sims[i] = eval_uts.IOU(mask1, mask2)

        return np.mean(sims)

    def merge_inst(self,
                   depth_in,
                   inst_id,
                   total_mask,
                   total_depth):
        """Merge the prediction of each car instance to a full image
        """

        render_depth = depth_in.copy()
        render_depth[render_depth <= 0] = np.inf
        depth_arr = np.concatenate([render_depth[None, :, :],
                                    total_depth[None, :, :]], axis=0)
        idx = np.argmin(depth_arr, axis=0)

        total_depth = np.amin(depth_arr, axis=0)
        total_mask[idx == 0] = inst_id

        return total_mask, total_depth

    def rescale(self, image, intrinsic):
        """resize the image and intrinsic given a relative scale
        """

        intrinsic_out = uts.intrinsic_vec_to_mat(intrinsic, self.image_size)
        hs, ws = self.image_size
        image_out = cv2.resize(image.copy(), (ws, hs))

        return image_out, intrinsic_out

    def showAnn(self, image_name, settings, save_dir, alpha=0.8):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        car_pose_file = '/media/SSD_1TB/ApolloScape/ECCV2018_apollo/train/'+'%s.json' % image_name
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        #intrinsic = self.dataset.get_intrinsic(image_name)
        ### we use only camera5 intrinsics
        intrinsic = self.dataset.get_intrinsic("Camera_5")
        self.intrinsic = uts.intrinsic_vec_to_mat(intrinsic)

        merged_image = image.copy()
        mask_all = np.zeros(image.shape)
        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            mask = self.render_car_cv2(car_pose['pose'], car_name, image)
            mask_all += mask

        mask_all = mask_all * 255 / mask_all.max()
        cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)

        # Save figure
        plt.close('all')
        fig = plt.figure(frameon=False)
        #fig.set_size_inches(image.shape[1]/10, image.shape[0]/10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(merged_image)

        save_set_dir = os.path.join(save_dir, settings)
        if not os.path.exists(save_set_dir):
            os.mkdir(save_set_dir)
        fig.savefig(os.path.join(save_dir, settings, image_name + '.png'), dpi=1)

        return image

    def findTrans(self, image_name):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        #intrinsic = self.dataset.get_intrinsic(image_name)
        ### we use only camera5 intrinsics
        intrinsic = self.dataset.get_intrinsic("Camera_5")
        self.intrinsic = uts.intrinsic_vec_to_mat(intrinsic)

        merged_image = image.copy()

        dis_trans_all = []
        for car_pose in car_poses:
            car_name = car_models.car_id2name[car_pose['car_id']].name

            car = self.car_models[car_name]
            pose = np.array(car_pose['pose'])
            # project 3D points to 2d image plane
            rmat = uts.euler_angles_to_rotation_matrix(pose[:3])

            x_y_z_R = np.matmul(rmat, np.transpose(np.float32(car['vertices'])))
            x_y_z = x_y_z_R + pose[3:][:, None]
            x2 = x_y_z[0]/x_y_z[2]
            y2 = x_y_z[1]/x_y_z[2]
            u = intrinsic[0] * x2 + intrinsic[2]
            v = intrinsic[1] * y2 + intrinsic[3]

            ###
            fx = intrinsic[0]
            fy = intrinsic[1]
            cx = intrinsic[2]
            cy = intrinsic[3]

            xc = ((u.max() + u.min())/2 - cx) / fx
            yc = ((v.max() + v.min())/2 - cy) / fy
            ymin = (v.min() - cy) / fy
            ymax= (v.max() - cy) / fy
            Rymin = x_y_z_R[1, :].min()
            Rymax = x_y_z_R[1, :].max()

            Rxc = x_y_z_R[0, :].mean()
            Ryc = x_y_z_R[1, :].mean()
            Rzc = x_y_z_R[2, :].mean()

            # Rxc = 0
            # Ryc = 0
            # Rzc = 0
            # Rxc = (x_y_z_R[0, :].max() + x_y_z_R[0, :].min())/2
            # Ryc = (x_y_z_R[1, :].max() + x_y_z_R[1, :].min())/2
            # Rzc = (x_y_z_R[2, :].max() + x_y_z_R[2, :].min())/2
            # Because the car highest point happened in the center!
            #zc = (Ryc - Rymin) / (yc - ymin)
            zc = (Ryc - Rymax) / (yc - ymax)

            xt = zc * xc - Rxc
            yt = zc * yc - Ryc
            zt = zc - Rzc
            pred_pose = np.array([xt, yt, zt])
            dis_trans = np.linalg.norm(pred_pose - pose[3:])

            # pose_pred_all = np.concatenate([car_pose['pose'][:3], pred_pose])
            # mask = self.render_car_cv2(pose_pred_all, car_name, image)
            # cv2.addWeighted(image.astype(np.uint8), 1.0, mask.astype(np.uint8), 0.5, 0, merged_image)
            # plt.imshow(merged_image)

            print(dis_trans)
            dis_trans_all.append(dis_trans)

        return dis_trans_all

        if False:
            xmin = (u.min() - cx) / fx
            xmax = (u.max() - cx) / fx
            ymin = (v.min() - cy) / fy
            ymax = (v.max() - cy) / fy

            Rxmin = x_y_z_R[0, :].min()
            Rxmax = x_y_z_R[0, :].max()
            Rymin = x_y_z_R[1, :].min()
            Rymax = x_y_z_R[1, :].max()
            Rzmin = x_y_z_R[2, :].min()
            Rzmax = x_y_z_R[2, :].max()

            # z1 = (Rxmax - Rxmin) / (xmax - xmin)
            # z2 = (Rymax - Rymin) / (ymax - ymin)
            #xt = (xmax*xmin) /(ymax*xmin-ymin*xmax) * (ymin*Rxmin/xmin - ymax*Rxmax/ymin - Rymin)
            xt = (Rxmax * xmin - Rxmin * xmax) / (xmax-xmin)
            yt = (Rymax * ymin - Rymin * ymax) / (ymax-ymin)

            ztxmin = (xt + Rxmin) /xmin - Rzmin
            ztxmax = (xt + Rxmax) / xmax - Rzmin
            ztymin = (yt + Rymin) / ymin - Rzmin
            ztymax = (yt + Rymax) / ymax - Rzmin

            pred_pose = np.array([xt, yt, ztymin])
            dis_trans = np.linalg.norm(pred_pose - pose[3:])

            pred_pose = np.array([xt, yt, ztxmin])
            dis_trans = np.linalg.norm(pred_pose - pose[3:])

    def findCarModels(self, image_name):
        """accumuate the areas of cars in an image
        Input:
            image_name: the name of image
        Output:

        """
        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        car_id = []
        for pose in car_poses:
            car_id.append(pose['car_id'])
            car_name = car_models.car_id2name[pose['car_id']].name

            if pose['car_id'] in self.car_counts.keys():
                self.car_counts[pose['car_id']]['car_counts']  += 1
            else:
                self.car_counts[pose['car_id']] = {}
                self.car_counts[pose['car_id']]['car_name'] = car_name
                self.car_counts[pose['car_id']]['car_counts'] = 0
        return car_id

    def findArea(self, image_name):
        """accumuate the areas of cars in an image
        Input:
            image_name: the name of image
        Output:

        """
        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        areas = []
        for pose in car_poses:
            areas.append(pose['area'])
        return areas

    def collect_pose(self, image_name):
        """accumuate the pose of cars in an image
        Input:
            image_name: the name of image
        Output:

        """
        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        areas = []
        for pose in car_poses:
            quaternions = uts.euler_angles_to_quaternions(np.array(pose['pose'][:3]))
            areas.append(np.concatenate((np.array(pose['pose']), quaternions[0])))
        return areas

    def showAnn_image(self, image_name, car_pose_file, settings, save_dir, alpha=0.8):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        with open(car_pose_file) as f:
            car_poses = json.load(f)
        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        #intrinsic = self.dataset.get_intrinsic(image_name)
        ### we use only camera5 intrinsics
        intrinsic = self.dataset.get_intrinsic("Camera_5")
        self.intrinsic = uts.intrinsic_vec_to_mat(intrinsic)
        merged_image = image.copy()
        mask_all = np.zeros(image.shape)
        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            mask = self.render_car_cv2(car_pose['pose'], car_name, image)
            mask_all += mask

        mask_all = mask_all * 255 / mask_all.max()
        cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), alpha, 0, merged_image)

        # Save figure
        plt.close('all')
        fig = plt.figure(frameon=False)
        fig.set_size_inches(image.shape[1]/100, image.shape[0]/100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(merged_image)
        save_set_dir = os.path.join(save_dir, settings)
        if not os.path.exists(save_set_dir):
            os.mkdir(save_set_dir)
        fig.savefig(os.path.join(save_dir, settings, image_name + '.png'), dpi=50)

        return image


class LabelResaver(object):
    """ Resave the raw labeled file to the required json format for evaluation
    """

    # (TODO Peng) Figure out why running pdb it is correct, but segment fault when
    # running
    def __init__(self, args):
        self.visualizer = CarPoseVisualizer(args, scale=0.5)
        self.visualizer.load_car_models()

    def strs_to_mat(self, strs):
        """convert str to numpy matrix
        """
        assert len(strs) == 4
        mat = np.zeros((4, 4))
        for i in range(4):
            mat[i, :] = np.array([np.float32(str_f) for str_f in strs[i].split(' ')])

        return mat

    def read_car_pose(self, file_name):
        """load the labelled car pose
        """
        cars = []
        lines = [line.strip() for line in open(file_name)]
        i = 0
        while i < len(lines):
            car = OrderedDict([])
            line = lines[i].strip()
            if 'Model Name :' in line:
                car_name = line[len('Model Name : '):]
                car['car_id'] = car_models.car_name2id[car_name].id
                pose = self.strs_to_mat(lines[i + 2: i + 6])
                pose[:3, 3] = pose[:3, 3] / 100.0  # convert cm to meter
                rot = uts.rotation_matrix_to_euler_angles(
                    pose[:3, :3], check=False)
                trans = pose[:3, 3].flatten()
                pose = np.hstack([rot, trans])
                car['pose'] = pose
                i += 6
                cars.append(car)
            else:
                i += 1

        return cars

    def convert(self, pose_file_in, pose_file_out):
        """ Convert the raw labelled file to required json format
        Input:
            file_name: str filename
        """
        car_poses = self.read_car_pose(pose_file_in)
        car_num = len(car_poses)
        MAX_DEPTH = self.visualizer.MAX_DEPTH
        image_size = self.visualizer.image_size
        intrinsic = self.visualizer.dataset.get_intrinsic(pose_file_in)
        self.visualizer.intrinsic = uts.intrinsic_vec_to_mat(intrinsic,
                                                             image_size)
        self.depth = MAX_DEPTH * np.ones(image_size)
        self.mask = np.zeros(self.depth.shape)
        vis_rate = np.zeros(car_num)

        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            depth, mask = self.visualizer.render_car(car_pose['pose'], car_name)
            self.mask, self.depth = self.visualizer.merge_inst(
                depth, i + 1, self.mask, self.depth)
            vis_rate[i] = np.float32(np.sum(mask == (i + 1))) / (np.float32(np.sum(mask)) + np.spacing(1))

        keep_idx = []
        for i, car_pose in enumerate(car_poses):
            area = np.round(np.float32(np.sum(self.mask == (i + 1))) / (self.visualizer.scale ** 2))
            if area > 1:
                keep_idx.append(i)

            car_pose['pose'] = car_pose['pose'].tolist()
            car_pose['area'] = int(area)
            car_pose['visible_rate'] = float(vis_rate[i])
            keep_idx.append(i)

        car_poses = [car_poses[idx] for idx in keep_idx]
        with open(pose_file_out, 'w') as f:
            json.dump(car_poses, f, sort_keys=True, indent=4,
                      ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render car instance and convert car labelled files.')
    parser.add_argument('--image_name', default='180116_053947113_Camera_5',
                        help='the dir of ground truth')
    parser.add_argument('--data_dir', default='../apolloscape/3d_car_instance_sample/',
                        help='the dir of ground truth')
    args = parser.parse_args()
    assert args.image_name

    print('Test converter')
    pose_file_in = './test_files/%s.poses' % args.image_name
    pose_file_out = './test_files/%s.json' % args.image_name
    label_resaver = LabelResaver(args)
    label_resaver.convert(pose_file_in, pose_file_out)

    print('Test visualizer')
    visualizer = CarPoseVisualizer(args)
    visualizer.load_car_models()
    visualizer.showAnn(args.image_name)
