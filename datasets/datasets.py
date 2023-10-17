import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import numpy as np
import os.path as path
import open3d as o3d
import utils.utils as utils
import MinkowskiEngine as ME
from diskcache import FanoutCache
import getpass
import tqdm
import time 
from voxel_hash_map.voxelize import voxel_down_sample

class UncuredFieldClouds(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.len = -1
        self.setup()
        self.loader = [ self.train_dataloader() ]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.cfg['mode'] == "pp":
            self.data_train = PostProcessingData(self.cfg['data']['path'])
        else:
            self.data_train = PreClusteredClouds(self.cfg['data']['path'])
        return

    def collate_nothing(self, data):
        return data

    def train_dataloader(self):
        loader = DataLoader(self.data_train, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers = self.cfg['train']['workers'],
                            shuffle= True,
                            collate_fn = self.collate_nothing)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


#################################################
################## Data loader ##################
#################################################

def read_data(file_name):
    cloud = o3d.io.read_point_cloud(file_name)
    return utils.SerializablePcd(cloud)

class PreClusteredClouds(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
        self.file_names = os.listdir(self.datapath)
        self.transform = utils.Transform()
        self.len = len(self.file_names)

    def __getitem__(self, index):
        cloud = read_data(os.path.join(self.datapath, self.file_names[index])).to_open3d() 
        view1 = self.transform(cloud)
        #view2 = self.transform(cloud)
        return view1 #, view2
    
    def __len__(self):
        return self.len

class PostProcessingData(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
        self.file_names = os.listdir(self.datapath)
        self.len = len(self.file_names)

    def __getitem__(self, index):
        cloud_name = os.path.join(self.datapath, self.file_names[index]) 
        cloud = o3d.t.io.read_point_cloud(cloud_name)
        points = cloud.point['positions'].numpy()[ (cloud.point['plant_ids'].numpy() != -1)[:,0] ]
        points, idxs = voxel_down_sample(points, 0.03)
        colors = cloud.point['colors'].numpy()[ (cloud.point['plant_ids'].numpy() != -1)[:,0] ]
        colors = colors[idxs, :]
        data = np.concatenate((points, colors),1)
        
        leaf_id = cloud.point['leaf_ids'].numpy()[ (cloud.point['plant_ids'].numpy() != -1)[:,0] ]
        return [ data , leaf_id[idxs,:] ]

    def __len__(self):
        return self.len



def array_to_sequence(batch_data):
    return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord/0.001), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]
    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
        
        return ME.SparseTensor(
                    features=p_feats,
                    coordinates=p_coord,
                    device=device), ME.SparseTensor(
                    features=p_label,
                    coordinates=p_coord,
                    device=device)

    return ME.SparseTensor(
            features=p_feats,
            coordinates=p_coord,
            device=device)


