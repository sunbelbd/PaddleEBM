import os 
import numpy as np
import scipy.io
from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register()
class VoxelDataSet(BaseDataset):
    """Import voxel from mat files.
    """
    def __init__(self, dataroot, data_size=100000, resolution=64, mode="train", category="modelnet10"):
        """Initialize this dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.

        """
        super(VoxelDataSet, self).__init__()
        self.dataset = self.load_data(dataroot, mode, category)
        self.dataset = self.dataset[:data_size].astype(np.float32)
        if resolution == 32: 
            self.dataset = self._down_sampling(self.dataset)
        elif resolution != 64:
            raise "Resolution should be 32 or 64"
        self.dataset = self._normalization(self.dataset)
        
    def load_data(self, dataroot, mode="train", category="modelnet10"):

        train_data = []
        if category == "modelnet40":
            categories = ['cup', 'bookshelf', 'lamp', 'stool', 'desk', 'toilet', 'night_stand', 'bowl', 'door', 'flower_pot', 'plant', 'stairs', 'bottle', 'mantel', 'sofa', 'laptop', 'xbox', 'tent', 'piano', 'car', 'wardrobe', 'tv_stand', 'cone', 'range_hood', 'bathtub', 'curtain', 'sink', 'glass_box', 'bed', 'chair', 'person', 'radio', 'dresser', 'bench', 'airplane', 'guitar', 'keyboard', 'table', 'monitor', 'vase']
            for cat in categories:
                with open(os.path.join(dataroot, "%s_%s_voxel.mat" % (cat, mode)), "rb") as f:
                    d = scipy.io.loadmat(f)["voxel"]
                train_data.append(d)
            train_data = np.concatenate(train_data)
        elif category == "modelnet10":
            categories = ['desk', 'toilet', 'night_stand', 'sofa', 'bathtub', 'bed', 'chair', 'dresser', 'table', 'monitor']
            for cat in categories:
                with open(os.path.join(dataroot, "%s_%s_voxel.mat" % (cat, mode)), "rb") as f:
                    d = scipy.io.loadmat(f)["voxel"]
                train_data.append(d)
            train_data = np.concatenate(train_data)
        else: 
            with open(os.path.join(dataroot, "%s_%s_voxel.mat" % (category, mode)), "rb") as f:
                train_data = scipy.io.loadmat(f)["voxel"]
        return train_data

    def _down_sampling(self, data): 

        import skimage.measure
        return skimage.measure.block_reduce(data, (1,2,2,2), np.max)
    
    
    def _normalization(self, data): 

        data_mean = data.mean()
        print("Perform normalization, mean = %.4f" % data_mean)
        return data - data_mean
