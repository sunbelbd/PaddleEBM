import os 
import numpy as np
import h5py
from .builder import DATASETS
from .base_dataset import BaseDataset


class PointCloudDataSet(BaseDataset):
    """Import pointcloud from obj files.
    """
    def __init__(self, dataroot, data_size=100000, num_point=2048, normalize_method="per_shape", data_argment=False, mode="train", category="modelnet10"):
        """Initialize this dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.

        """
        super(PointCloudDataSet, self).__init__()
        self.num_point = num_point
        self.dataset = self.load_data(dataroot, mode, category)
        self.dataset = self.dataset[:data_size].astype(np.float32)
        if normalize_method:
            self.dataset = self._normalization(self.dataset, normalize_method)
        if data_argment:
            self.dataset = self._data_argment(self.dataset)

    @staticmethod
    def load_data(self, dataroot, mode, category):
        raise NotImplementedError

    def _normalization(self, data, normalize_method): 

        # Normalize to [-1, 1]
        if normalize_method == 'ebp':
            max_val = data.max()
            min_val = data.min()
            data = ((data - min_val) / (max_val - min_val)) * 2 - 1
        elif normalize_method == 'scale': 
            data = data - data.data(axis=(0,1))
            max_val = np.abs(data).max(axis=(0,1))
            train_data = data / max_val 
        elif normalize_method == 'per_shape': 
            mean = np.mean(data[..., :3], axis=1, keepdims=True).repeat(data.shape[1], 1)
            data -= mean 
            data  = data / np.max(np.abs(data), axis=(1, 2), keepdims=True)
        return data 

    def _data_argment(self, data):
        # Enriched Normailzation 
        print("Argment data by rotating through x axis. %d Data" % len(data))

        # Rotate through X-axis 
        out_data = []
        from utils.eulerangles import euler2mat
        import math
        for xrot in (np.arange(-1, 1, 0.25) * math.pi):
            M = euler2mat(0, xrot, 0)
            out_data.append(np.dot(data, M.transpose()))
        print("Argmented. %d Data" % len(data))
        return np.concatenate(out_data)

    def preprocess(self, data): 
        
        if len(data) >= self.num_point: 
            idx = np.random.permutation(len(data))[:self.num_point]
        else:
            idx = np.concatenate([np.arange(len(data)), 
            np.random.choice(np.arange(len(data)), size=self.num_point - len(data), replace=True)])
        return np.swapaxes(data[idx], -1, -2) 


@DATASETS.register()
class ModelNetPreprocessedPointCloud(PointCloudDataSet): 

    def load_data(self, dataroot, mode="train", category="modelnet10"):

        if category == "modelnet40":
            train_data = []
            categories = ['cup', 'bookshelf', 'lamp', 'stool', 'desk', 'toilet', 'night_stand', 'bowl', 'door', 'flower_pot', 'plant', 'stairs', 'bottle', 'mantel', 'sofa', 'laptop', 'xbox', 'tent', 'piano', 'car', 'wardrobe', 'tv_stand', 'cone', 'range_hood', 'bathtub', 'curtain', 'sink', 'glass_box', 'bed', 'chair', 'person', 'radio', 'dresser', 'bench', 'airplane', 'guitar', 'keyboard', 'table', 'monitor', 'vase']
            for cat in categories:
                d = np.load(os.path.join(dataroot, "%s_%s.npy" % (cat, mode)))
                train_data.append(d)
            train_data = np.concatenate(train_data)
        elif category == "modelnet10":
            train_data = []
            categories = ['desk', 'toilet', 'night_stand', 'sofa', 'bathtub', 'bed', 'chair', 'dresser', 'table', 'monitor']
            for cat in categories:
                d = np.load(os.path.join(dataroot, "%s_%s.npy" % (cat, mode)))
                train_data.append(d)
            train_data = np.concatenate(train_data)
        else: 
            train_data = np.load(os.path.join(dataroot, "%s_%s.npy" % (category, mode)))
        return train_data

@DATASETS.register()
class PartNetPointCloud(PointCloudDataSet): 

    def load_data(self, dataroot, mode, category):

        # partnet_base = "/home/fei960922/Documents/Dataset/PointCloud/partnet/sem_seg_h5"
        if category == "all":
            categories = os.listdir(dataroot)
        else: 
            categories = [category]
        train_data = []
        for cat in categories:
            for name in [s for s in os.listdir("%s/%s" % (dataroot, cat)) 
            if s[:3]=="tra" and s[-1]=="5"]:
                h5file = h5py.File("%s/%s/%s" % (dataroot, cat, name))
                d = np.asarray(h5file['data'])
                # seg = h5file['label_seg']
                train_data.append(d)
        return train_data

@DATASETS.register()
class ShapenetPointCloud(PointCloudDataSet): 

    def load_data(self, dataroot, mode, category):

        train_data = np.load(os.path.join(dataroot, "shapenet15k_%s_%s.npy" % (category, self.mode)))
        return train_data

# TODO: Support ShapeNetPart, Mnist etc.
@DATASETS.register()
class OtherPointCloud(PointCloudDataSet): 

    def __init__(self, dataroot, mode, category):

        super(OtherPointCloud, self).__init__(dataroot, mode)
        raise NotImplementedError

        # elif cate_temp == "shapenetpart":
        #     pcd, label, seg = [], [], []
        #     for i in range(10):
        #         try:
        #             f = h5py.File("data/hdf5_data/ply_data_train%d.h5" % i, 'r')
        #             pcd.append(f['data'][:])
        #             label.append(f['label'][:])
        #             seg.append(f['pid'][:])
        #         except:
        #             break 
        #     train_data, train_label, train_seg = np.concatenate(pcd), np.concatenate(label), np.concatenate(seg)

        #     # data stored in list as a pickle file. range from [-1, 1]
        #     # path = "data/shapenetpart_training.pkl" 
        #     # with open(path, "rb") as f: 
        #     #     data = pickle.load(f)
        #     if len(category.split("_")) > 1:
        #         idx = int(category.split("_")[1])
        #         print((train_label == idx).shape)
        #         train_data = train_data[(train_label == idx).squeeze()]

        # elif cate_temp == "mnist":
            
        #     train_data = pickle.load(open("data/mnist_normal.pkl", "rb"))
        #     # for pnt in temp_data:
        #     #     train_data.append(np.concatenate([pnt, np.zeros(([pnt.shape[0], 1]))], 1))
        # else: 
        #     train_data = [np.load('%s/%s_train.npy' % (config.data_path, category))]
        #     print('%s/%s_train.npy loaded' % (config.data_path, category))
        
