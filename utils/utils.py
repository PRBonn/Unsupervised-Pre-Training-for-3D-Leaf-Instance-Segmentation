import random
import yaml
import numpy as np
import copy
import open3d as o3d
import torch
import time
import voxel_hash_map
from voxel_hash_map import voxelize

class RandomErasing3D:
    def __init__(self, probability, points_ratio):
        self.p = probability
        self.ratio = points_ratio
    
    def __call__(self, pcd):
        if random.random() < self.p:
            pnts = np.asarray(pcd.points)
            clrs = np.asarray(pcd.colors)
            N = len(pnts)
            N_new = int(N * (1 - self.ratio))
            if N_new == N: return pcd

            choice = np.zeros((N), dtype=np.bool)
            choice[:N_new] = 1
            np.random.shuffle(choice)
            
            pnts = pnts[choice]
            clrs = clrs[choice]

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pnts)
            cloud.colors = o3d.utility.Vector3dVector(clrs)
            return cloud
        return pcd

class ChromaticTranslation:
    def __init__(self, probability, max_color_shift):
        self.p = probability
        self.max_color_shift = max_color_shift

    def __call__(self, colors):
        if random.random() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 2 * self.max_color_shift
            return np.clip(tr + colors, 0, 1)
        return colors

class GaussianNoise:
    def __init__(self, probability, max_magnitude, sigma):
        self.p = probability
        self.max_magnitude = max_magnitude 
        self.sigma = sigma

    def __call__(self, pnts):
        if random.random() < self.p:
            noise = self.sigma * np.random.randn(pnts.shape[0], pnts.shape[1])
            noise = noise.clip(-self.max_magnitude, self.max_magnitude)
            return pnts + noise
        return pnts

class RotoTranslation3D:
    def __init__(self, probability, max_t, max_alpha, max_beta, max_gamma):
        self.p = probability
        self.max_t = max_t
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.max_gamma = max_gamma

    def __call__(self, pnts):
        if random.random() < self.p:
            # sample randomly translation on x, y, z 
            translate = np.random.rand(1,3) * self.max_t

            # sample three rotation angles
            angles = np.random.rand(1,3) * [ self.max_alpha, self.max_beta, self.max_gamma ]
            
            # build rotation matrix
            Rx = [ [1, 0 , 0] , [ 0 , np.cos(angles[0,0]) , - np.sin(angles[0,0]) ] , [ 0 , np.sin(angles[0,0]), np.cos(angles[0,0]) ] ] 
            Ry = [ [ np.cos(angles[0,1]) , 0 , np.sin(angles[0,1]) ] , [0 ,1 ,0 ] , [ -np.sin(angles[0,1]), 0 , np.cos(angles[0,1]) ] ] 
            Rz = [ [ np.cos(angles[0,2]) , - np.sin(angles[0,2]) ,0 ] , [np.sin(angles[0,2]), np.cos(angles[0,2]), 0 ] , [ 0 , 0 ,1 ] ] 
            
            # construct 4 x 4 homogeneous transformation matrix
            T = np.eye(4)
            T[:3,:3] = np.asarray(Rz) @ np.asarray(Ry) @ np.asarray(Rx) 
            T[:3,3] = translate
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pnts)
            pcd = pcd.transform(T)
            return np.asarray(pcd.points)
        return pnts

class Distortion:
    def __init__(self, probability, max_angles):
        self.p = probability
        self.max_angles = max_angles

    def __call__(self, points):
        if random.random() > self.p: 
            return points
        
        max_distortion_angles = (np.random.rand(1,3) * self.max_angles)[0]
        points -= np.mean(points, axis=0) 

        # compute distance from mean, which is now 0
        d = ((points**2).sum(axis=1))**0.5
        
        # compute angles for each point
        alphas = max_distortion_angles[0] * d
        betas = max_distortion_angles[1] * d
        gammas = max_distortion_angles[2] * d

        R = np.array([ [ np.cos(betas)*np.cos(gammas) , np.sin(alphas)*np.sin(betas)*np.cos(gammas) - np.cos(alphas)*np.sin(gammas) , np.cos(alphas)*np.sin(betas)*np.cos(gammas) + np.sin(alphas)*np.sin(gammas) ] , [ np.cos(betas)*np.sin(gammas) , np.sin(alphas)*np.sin(betas)*np.sin(gammas) + np.cos(alphas)*np.cos(gammas) , np.cos(alphas)*np.sin(betas)*np.sin(gammas) - np.sin(alphas)*np.cos(gammas)] , [ - np.sin(betas) , np.sin(alphas)*np.cos(betas) , np.cos(alphas)*np.cos(betas)] ])
        R = torch.tensor(R).cuda()
        try:
            points = torch.tensor(points).cuda()
            new_points = (R.T @ points.T)[torch.arange(0, len(points)).cuda(),:, torch.arange(0, len(points)).cuda()]
        except RuntimeError:
            print("point cloud is too big for distortion")
            return points.cpu()
        return np.asarray(new_points.cpu())

 
class LeavesOcclusion:
    def __init__(self, probability, leaf_lenght, leaf_width):
        self.p = probability
        self.len = leaf_lenght
        self.width = leaf_width

    def __call__(self, points, colors):
        if random.random() > self.p or points.shape[0] == 0: 
            return points, colors     
        # extract borders to put the ellipse center on the edge of the plant
        borders_max = np.max(points, axis=0)
        borders_min = np.min(points,0)
        # two random ellipse centers
        index_max, index_min = random.randint(0,1), random.randint(0,1)

        center_max = points[ np.abs(points[:,index_max] - borders_max[index_max]) < 1e-2 ][ np.random.randint(0, points[ np.abs(points[:,index_max] - borders_max[index_max]) < 1e-2 ].shape[0] ) ]
        center_min = points[ np.abs(points[:,index_min] - borders_min[index_min]) < 1e-2 ][ np.random.randint(0, points[ np.abs(points[:,index_min] - borders_min[index_min]) < 1e-2 ].shape[0] ) ]

        # random sizes for the ellipse's axes
        axis_0_x, axis_0_y, axis_1_x, axis_1_y = random.random()*self.width, random.random()*self.len, random.random()*self.width, random.random()*self.len 

        # condition of erasing: all points outside of two ellipses are kept
        ellipse_0 = (points[:,0] - center_max[0])**2/axis_0_x**2 + (points[:,1] - center_max[1])**2/axis_0_y**2 >= 1
        if ellipse_0.sum() > 5000:
            points = points[ ellipse_0 ]
            colors = colors[ ellipse_0 ]

        ellipse_1 = (points[:,0] - center_min[0])**2/axis_1_x**2 + (points[:,1] - center_min[1])**2/axis_1_y**2 >= 1
        if ellipse_1.sum() > 5000:
            points = points[ ellipse_1 ]
            colors = colors[ ellipse_1 ]

        return points, colors

               

class CloudToInput:
    def __init__(self):
        pass

    def __call__(self, pnts, clrs):
        return np.concatenate( (pnts, clrs), -1) 

class Transform:
    def __init__(self):
        
        self.random_erasing = RandomErasing3D(probability = 0.0,
                                              points_ratio = 0.3)

        self.distortion = Distortion(probability = 0.8, max_angles = np.asarray([np.pi, np.pi, np.pi]))
        
        self.occlusion = LeavesOcclusion(probability = 0.0, leaf_lenght = 0.1, leaf_width = 0.04)

        self.chromatic_translation = ChromaticTranslation(probability = 0.8,
                                                          max_color_shift = 0.125)

        self.gaussian_noise = GaussianNoise(probability = 1.0,
                                            max_magnitude = 0.005,
                                            sigma = 0.002)
        
        self.rotate_and_translate = RotoTranslation3D(probability = 0.8,
                                                      max_t = 3,
                                                      max_alpha = np.pi,
                                                      max_beta = np.pi,
                                                      max_gamma = np.pi)
            
        self.to_input = CloudToInput()

    def __call__(self, x):
        # create first augmentation, keep indeces of saved points
        view1 = self.random_erasing(x)
        pnts, idxs = voxelize.voxel_down_sample(np.asarray(view1.points), 0.015)
        pnts = self.distortion(pnts)
        pnts, colors = self.occlusion(pnts, np.asarray(x.colors)[idxs,:])
        colors = self.chromatic_translation(colors)
        pnts = self.gaussian_noise(pnts)
        pnts = self.rotate_and_translate(pnts)
        net_input = self.to_input(pnts, colors)
        return net_input

class SerializablePcd:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.points = np.asarray(pcd.points)
        self.normals = np.asarray(pcd.normals)
        self.colors = np.asarray(pcd.colors)

    def to_open3d(self) -> o3d.geometry.PointCloud:

        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(self.points))
        )
        pcl.normals=o3d.utility.Vector3dVector(np.asarray(self.normals))
        pcl.colors=o3d.utility.Vector3dVector(np.asarray(self.colors))
        return pcl


