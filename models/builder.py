#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import paddle

from utils.registry import Registry
from .modules.init import init_weights

MODELS = Registry("MODEL")
NETWORKS = Registry("NETWORK")
SAMPLINGS = Registry("SAMPLING")

def build_model(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name', None)
    model = MODELS.get(name)(**cfg_)
    return model

def get_network(cfg, network_type=""):
     
    if cfg is None: 
        return None
    cfg_ = cfg.copy()
    net_name = cfg_.pop('name', None).capitalize()
    if net_name is None or net_name == "None":
        return None
    else: 
        netG = NETWORKS.get(network_type + net_name)(**cfg_)
        if getattr(cfg_, "use_default_init", False):
            init_cfgs = {k: v for k, v in cfg_.items() if k in [
                'init_type', 'init_gain', 'distribution']}
            init_weights(netG, **init_cfgs)
        return netG

def get_sampling(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name', None)
    sampling = SAMPLINGS.get("Sampling" + name)(**cfg_)
    return sampling