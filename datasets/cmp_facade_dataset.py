#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import cv2
import os.path
import numpy as np
import paddle
from .base_dataset import BaseImageDataset
from .image_folder import ImageFolder

from .builder import DATASETS
from .transforms.builder import build_transforms


@DATASETS.register()
class CmpFacadeDataset(paddle.io.Dataset):
    """
    """
    def __init__(self,
                 dataroot="data/CmpFacade",
                 transforms=None, 
                 split="base"):
        """Initialize this dataset class.

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        self.x_root = os.path.join(dataroot, split, 'photos')
        self.y_root = os.path.join(dataroot, split, 'facade')
        self.x = ImageFolder(self.x_root, transform=build_transforms(transforms), loader=self.loader)
        self.y = ImageFolder(self.y_root, transform=build_transforms(transforms), loader=self.loader)
        assert len(self.x) == len(self.y), "Length of the paired data unmatch."
    @staticmethod
    def loader(path):
        return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB).astype(np.float32)
    def __getitem__(self, index):
        return {
            'data': self.x[index].transpose([2, 0, 1]) / 255 * 2 - 1,
            'class_id': self.y[index].transpose([2, 0, 1]) / 255 * 2 - 1,
        }

    def __len__(self):
        return len(self.x)
