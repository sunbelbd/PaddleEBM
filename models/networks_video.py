import paddle
import paddle.nn as nn

from .builder import NETWORKS
        
@NETWORKS.register()
class EnergyBasedVideonet(nn.Layer):
    """ Video structure used in STGConvNet
    """

    def __init__(self, **kwargs):
        """Construct the energy function used in GVoxelNet
        Args:
            None
        """
        super().__init__()
        net = []
        net.append(nn.Conv3D(3, 120, 7, padding="same", stride=3))
        net.append(nn.ReLU())
        net.append(nn.Conv3D(120, 30, [4, 75, 75], padding=[2,0,0], stride=[2, 1, 1]))
        net.append(nn.ReLU())
        net.append(nn.Conv3D(30, 5, [2, 1, 1], padding=[1,0,0], stride=1))
        net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


