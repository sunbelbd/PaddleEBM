import os 
import numpy as np
import h5py
import imageio
import cv2
from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register()
class VideoDataSet(BaseDataset):
    """Import pointcloud from obj files.
    """
    def __init__(self, dataroot, num_frames=70, image_size=224, num_channel=3, category="fire_pot"):
        """Initialize this dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.

        """
        super(VideoDataSet, self).__init__()
        datapath = os.path.join(dataroot, category)
        videos = [f for f in os.listdir(datapath) if f.endswith(".avi") or f.endswith(".mp4")]
        num_videos = len(videos)
        self.dataset = np.zeros((num_videos, num_frames, image_size, image_size, num_channel))
        self.mean_offset = np.zeros(num_videos)
        for i, path in enumerate(videos):
            vid = imageio.get_reader(os.path.join(datapath, path),  'ffmpeg')
            vid_array = np.stack([cv2.resize(a, (image_size, image_size)) for a in vid.iter_data()][:num_frames]).astype(np.float32)
            self.mean_offset[i] = vid_array.mean()
            self.dataset[i] = vid_array - self.mean_offset[i]
        self.dataset = np.transpose(self.dataset, [0, 4, 1, 2, 3])

@DATASETS.register()
class ImageDataSet(BaseDataset):
    """Import image from a folder.
    """
    def __init__(self, dataroot, image_size, category):
        """Initialize this dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.

        """
        super(ImageDataSet, self).__init__()
        datapath = os.path.join(dataroot, category)
        paths = [f for f in os.listdir(datapath) if f.endswith(".png") or f.endswith(".jpg")]
        self.dataset = np.zeros((len(paths), image_size, image_size, 3))
        for i, path in enumerate(paths):
            img = cv2.imread(os.path.join(datapath, path))
            self.dataset[i] = cv2.resize(img, (image_size, image_size)).astype(np.float32)
        self.mean_offset = self.dataset.mean()
        self.dataset = self.dataset - self.mean_offset
        self.dataset = np.transpose(self.dataset, [0, 3, 1, 2])
