import paddle
import paddle.nn as nn

from .builder import NETWORKS

class residual_Conv1d(nn.Layer):

    def __init__(self, h):
        super().__init__()
        self.layer = nn.Conv1D(h, h, 1) 
    def forward(self, x):
        return self.layer(x) + x

class residual_Linear(nn.Layer):

    def __init__(self, h):
        super().__init__()
        self.layer = nn.Linear(h, h) 
    def forward(self, x):
        return self.layer(x) + x

@NETWORKS.register()
class EnergyBasedGpointnet(nn.Layer):
    """ PointNet structure used in GPointNet
    """

    def __init__(self, hidden_size, batch_norm="LayerNorm", point_dim=3, num_point=2048, activation='ReLU',
    **kwargs):
        """Construct a EnergyBasedNetworkConv2D
        Args:
            hidden_size (list<int>)      -- the number of hidden size for layers
            num_point (int)      -- number of points
            point_dim (int)     -- number of point dimension
            batch_norm (str)           -- normalization layer
            activation (str)          -- activation function (in paddle.nn)
        """
        super(EnergyBasedGpointnet, self).__init__()
        net_local, net_global = [], [] 
        prev = point_dim
        for h in hidden_size[0]: 
            layer = residual_Conv1d(h) if h==prev else nn.Conv1D(prev, h, 1) 
            # # Inherit from tf
            # nn.init.normal_(layer.weight, 0, 0.02) 
            # nn.init.zeros_(layer.bias) 
            net_local.append(layer)
            if batch_norm == "bn": 
                net_local.append(nn.BatchNorm1d(h)) # Question exists
            elif batch_norm == "ln": 
                net_local.append(nn.LayerNorm(num_point))
            elif batch_norm == "ln01": 
                net_local.append(nn.LayerNorm(num_point, elementwise_affine=False))
            elif batch_norm == "lnm": 
                net_local.append(nn.LayerNorm([h, num_point], elementwise_affine=False))
            elif batch_norm == "in": 
                net_local.append(nn.InstanceNorm1d(h)) # Question exists
            else: 
                raise NotImplementedError
            if activation != "":
                net_local.append(getattr(paddle.nn, activation)())
            prev = h
        for h in hidden_size[1]: 
            layer = residual_Linear(h) if h==prev else nn.Linear(prev, h) 
            net_global.append(layer)
            if activation != "":
                net_global.append(getattr(paddle.nn, activation)())
            prev = h
        net_global.append(nn.Linear(prev, 1))
        self.local = nn.Sequential(*net_local)
        self.globals = nn.Sequential(*net_global)
    
    def forward(self, point_cloud, out_local=False, out_every_layer=False):

        if out_local: 
            if out_local == True:
                output_local = len(self.local) - 1
            for i, layer in enumerate(self.local): 
                point_cloud = layer(point_cloud)
                if i == out_local:
                    local = point_cloud 
            return paddle.mean(point_cloud, 2), local
        local = self.local(point_cloud)
        out = self.globals(paddle.mean(local, 2))
        return out 

    def _output_all(self, pcs):
        
        res = [] 
        for layer in self.local: 
            pcs = layer(pcs) 
            if type(layer) is nn.LayerNorm: 
                res.append(pcs)
        return res 

        
@NETWORKS.register()
class EnergyBasedVoxelnet(nn.Layer):
    """ VoxelNet structure used in VoxelNet
    """

    def __init__(self, **kwargs):
        """Construct the energy function used in GVoxelNet
        Args:
            None
        """
        
        super().__init__()
        net = []
        net.append(nn.Conv3D(1, 200, 16, stride=3, padding="same"))
        net.append(nn.ReLU())
        net.append(nn.Conv3D(200, 100, 6, stride=2, padding="same"))
        net.append(nn.ReLU())
        net.append(nn.Flatten())
        net.append(nn.Linear(100 * 6 * 6 * 6, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x.unsqueeze(1))


