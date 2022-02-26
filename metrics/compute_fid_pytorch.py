
import torch
import paddle
import numpy as np
import pytorch_fid_wrapper as pfw

class Fid_calculator(object):

    def __init__(self, training_data):
        pfw.set_config(batch_size=100, device=torch.device('cuda'))
        training_data = training_data.repeat(1,3 if training_data.shape[1] == 1 else 1,1,1)
        print("precalculate FID distribution for training data...")
        self.real_m, self.real_s = pfw.get_stats(training_data)

    def fid(self, data): 
        if type(data) is np.ndarray:
            data = torch.Tensor(data)
        data = data.repeat(1,3 if data.shape[1] == 1 else 1,1,1) 
        return pfw.fid(data, real_m=self.real_m, real_s=self.real_s)

def make_fid_calculator(data_loader): 
    ds_fid = paddle.concat([x['data'] for x in data_loader]).numpy()  
    print(ds_fid.shape)  
    return Fid_calculator(torch.Tensor(ds_fid))