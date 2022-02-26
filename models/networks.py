import paddle
import paddle.nn as nn
import functools

from paddle.nn import BatchNorm2D
from .modules.norm import build_norm_layer
from .builder import NETWORKS


class build_residual_block(nn.Layer):

    def __init__(self, dim, norm_layer, use_dropout=False):
    
        super(build_residual_block, self).__init__()
        res = [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(dim, dim, 3, 1, 'valid'), 
            norm_layer(dim), 
            nn.ReLU()]
        if use_dropout: 
            res.append(nn.Dropout2D())
        res.append(nn.Pad2D([1, 1, 1, 1], mode='reflect'))
        res.append(nn.Conv2D(dim, dim, 3, 1, 'valid'))
        res.append(norm_layer(dim))
        self.net = nn.Sequential(*res)
    
    def forward(self, inputs):
        return inputs + self.net(inputs)

@NETWORKS.register()
class GeneratorImg2img(nn.Layer):
    """ Deep Convolutional Generator for MNIST
    """

    def __init__(self,
                 input_nz,
                 output_nc,
                 input_ncondition=0,
                 ngf=64,
                 norm_type='instance',
                 **kwargs):
        """Construct a GeneratorConv2D generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of dimension of the condition
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(GeneratorImg2img, self).__init__()

        self.input_nc = input_ncondition
        num_blocks = 9
        norm_layer = build_norm_layer(norm_type)

        net = [
            nn.Pad2D([3, 3, 3, 3], mode='reflect'),
            nn.Conv2D(3, 32, 7, 1, 'valid'), 
            norm_layer(32),
            nn.ReLU(), 
            nn.Conv2D(32, 64, 3, 2, 'same'), 
            norm_layer(64),
            nn.ReLU(), 
            nn.Conv2D(64, 128, 3, 2, 'same'), 
            norm_layer(128),
            nn.ReLU()]
        for r in range(num_blocks):
            net.append(build_residual_block(128, norm_layer, use_dropout=True))

        net += [
            nn.Conv2DTranspose(128, 64, 3, 2, 0), #??? 
            norm_layer(64),
            nn.ReLU(), 
            nn.Conv2DTranspose(64, 32, 3, 2, 0), #??? 
            norm_layer(32),
            nn.ReLU(), 
            nn.Pad2D([1, 2, 2, 1], mode='reflect'),
            nn.Conv2D(32, 3, 7, 1, 'valid'), 
            nn.Tanh(), 
        ]

        self.model = nn.Sequential(*net)

    def forward(self, z, condition):
        """Standard forward"""
        return self.model(condition)

@NETWORKS.register()
class EnergyBasedImg2img(paddle.nn.Layer):
    """ <A Theory of Generative ConvNet> Experiment 1
    """
    def __init__(self,
                 input_sz,
                 input_nc,
                 output_nc,
                 input_nz=0, 
                 nef=64,
                 **kwargs):
        super(EnergyBasedImg2img, self).__init__()

        model = [
            nn.Conv2D(input_nc + input_nz, nef, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2D(nef, nef * 2, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2D(nef * 2, nef * 4, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2D(nef * 4, nef * 8, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2D(nef * 8, 1, 6, 1)
        ]
        self.conv = nn.Sequential(*model)
        # self.fc = nn.Linear(nef * 8, 100)

    def forward(self, x, condition=None):
        """Standard forward"""
        if condition is not None: 
            inp = paddle.concat([x, condition])
        else: 
            inp = x
        return self.conv(inp).squeeze()
        

@NETWORKS.register()
class EnergyBasedImageshortrun(nn.Layer):
    """ Vanilla Deep Convolutional EBM used in short-run MCMC
    """

    def __init__(self, input_sz=32, input_nc=3, output_nc=1, nef=64, **kwargs):
        """Construct a EnergyBasedConv2d
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(EnergyBasedImageshortrun, self).__init__()

        negative_slope = 0.2
        self.conv = nn.Sequential(
            nn.Conv2D(input_nc, nef, 3, 1, 1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2D(nef, nef*2, 4, 2, 1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2D(nef*2, nef*4, 4, 2, 1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2D(nef*4, nef*8, 4, 2, 1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2D(nef * 8, 1, 4, 1, 0))

    def forward(self, x):
        """Standard forward"""
        return self.conv(x).squeeze()

@NETWORKS.register()
class EnergyBasedImagetexture(nn.Layer):
    """ <A Theory of Generative ConvNet> Experiment 1
    """
    def __init__(self, **kwargs):
        super(EnergyBasedImagetexture, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(3, 100, 15, padding="same", stride=3), nn.ReLU(),
            nn.MaxPool2D(3), 
            nn.Conv2D(100, 64, 5, padding="same", stride=2), nn.ReLU(),
            nn.MaxPool2D(2), 
            nn.Conv2D(64, 30, 5, padding="same", stride=2), nn.ReLU(),
            nn.Conv2D(30, 1, 3))
    def forward(self, x):
        return self.conv(x)

@NETWORKS.register()
class EnergyBasedImageobject(nn.Layer):
    """ <A Theory of Generative ConvNet> Experiment 1
    """
    def __init__(self, **kwargs):
        super(EnergyBasedImageobject, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(3, 100, 7, padding="same", stride=2), nn.LeakyReLU(),
            nn.Conv2D(100, 64, 5, padding="same", stride=2), nn.LeakyReLU(),
            nn.Conv2D(64, 20, 3, padding="same", stride=2), nn.LeakyReLU(),
            nn.Conv2D(20, 1, 10))
    def forward(self, x):
        return self.conv(x)

@NETWORKS.register()
class EnergyBasedConv2d(nn.Layer):
    """ Vanilla Deep Convolutional EBM
    """

    def __init__(self,
                 input_sz,
                 input_nc,
                 output_nc,
                 input_nz=0, 
                 nef=64,
                 **kwargs):
        """Construct a EnergyBasedConv2d
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(EnergyBasedConv2d, self).__init__()

        norm_layer = build_norm_layer("batch")
        pre_condition_net = [
            nn.Conv2DTranspose(input_nz, nef * 4, 4, 1, 0),
            norm_layer(nef * 4),
            nn.ReLU(),
            nn.Conv2DTranspose(nef * 4, nef * 2, 4, 1, 0),
            norm_layer(nef * 2),
            nn.ReLU(),
            nn.Conv2DTranspose(nef * 2, nef * 1, 4, 2, 1),
            norm_layer(nef * 1),
            nn.ReLU(),
            nn.Conv2DTranspose(nef * 1, input_nz, 4, 2, 1),
            nn.Tanh()
        ]
        model = [
            nn.Conv2D(input_nc + input_nz, nef, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2D(nef, nef * 2, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2D(nef * 2, nef * 4, 4, 2, 1),
            nn.LeakyReLU()
        ]

        self.conv = nn.Sequential(*model)
        self.pre_condition_net = nn.Sequential(*pre_condition_net)
        out_sz = input_sz // 8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_sz * out_sz * nef * 4, output_nc)
        )

    def forward(self, x, condition=None):
        """Standard forward"""
        if condition is not None: 
            cond_x = self.pre_condition_net(condition)
            inp = paddle.concat([x, cond_x])
        else: 
            inp = x
        return self.fc(self.conv(inp))


@NETWORKS.register()
class EncoderConv2d(nn.Layer):
    """ Vanilla Deep Convolutional Encoder
    """

    def __init__(self,
                 input_sz,
                 input_nc,
                 output_nc,
                 nef=64,
                 norm_type='batch',
                 **kwargs):
        """Construct a EncoderConv2D
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(EncoderConv2d, self).__init__()

        model = [
            nn.Conv2D(input_nc, nef, 4, 2, 1),
            nn.LeakyReLU(),

            nn.Conv2D(nef, nef, 4, 2, 1),
            nn.LeakyReLU(),

            nn.Conv2D(nef, nef, 4, 2, 1),
            nn.LeakyReLU()
        ]

        self.conv = nn.Sequential(*model)
        out_sz = input_sz // 8
        self.fc = nn.Linear(out_sz * out_sz * nef, output_nc)
        self.fcVar = nn.Linear(out_sz * out_sz * nef, output_nc)

    def forward(self, x):
        """Standard forward"""
        out = self.conv(x)
        out_flat = paddle.flatten(out, start_axis=1)
        fc = self.fc(out_flat)
        fcVar = self.fcVar(out_flat)

        return fc, fcVar


@NETWORKS.register()
class GeneratorCifar_coopvaebm(nn.Layer):
    """ Deep Convolutional Generator for Cifar
    """

    def __init__(self,
                 input_nz,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='batch',
                 **kwargs):
        """Construct a GeneratorConv2D generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(GeneratorMnist, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        model = [
            nn.Conv2DTranspose(input_nz, ngf * 4, 4, 1, 0),
            norm_layer(ngf * 4),
            nn.ReLU(),

            nn.Conv2DTranspose(ngf * 4, ngf * 2, 4, 1, 0),
            norm_layer(ngf * 2),
            nn.ReLU(),

            nn.Conv2DTranspose(ngf * 2, ngf * 1, 4, 2, 1),
            norm_layer(ngf * 1),
            nn.ReLU(),

            nn.Conv2DTranspose(ngf * 1, output_nc, 4, 2, 1),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)


@NETWORKS.register()
class GeneratorMnist(nn.Layer):
    """ Deep Convolutional Generator for MNIST
    """

    def __init__(self,
                 input_nz,
                 output_nc,
                 input_ncondition=0,
                 ngf=64,
                 norm_type='batch',
                 **kwargs):
        """Construct a GeneratorConv2D generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of dimension of the condition
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(GeneratorMnist, self).__init__()

        self.input_nc = input_ncondition
        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        model = [
            nn.Conv2DTranspose(input_nz + input_ncondition, ngf * 4, 4, 1, 0),
            norm_layer(ngf * 4),
            nn.ReLU(),
            nn.Conv2DTranspose(ngf * 4, ngf * 2, 4, 1, 0),
            norm_layer(ngf * 2),
            nn.ReLU(),
            nn.Conv2DTranspose(ngf * 2, ngf * 1, 4, 2, 1),
            norm_layer(ngf * 1),
            nn.ReLU(),
            nn.Conv2DTranspose(ngf * 1, output_nc, 4, 2, 1),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, condition=None):
        """Standard forward"""
        return self.model(x if condition is None else paddle.concat([x, condition.reshape((-1, self.input_nc, 1, 1))], axis=1))


@NETWORKS.register()
class GeneratorConv2d(nn.Layer):
    """ Deep Convolutional Generator
    """

    def __init__(self,
                 input_nz,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='batch',
                 **kwargs):
        """Construct a GeneratorConv2D generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(GeneratorConv2d, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        mult = 4
        n_downsampling = 3

        if norm_type == 'batch':
            model = [
                nn.Conv2DTranspose(input_nz,
                                   ngf * mult,
                                   kernel_size=4,
                                   stride=1,
                                   padding=0,
                                   bias_attr=use_bias),
                BatchNorm2D(ngf * mult),
                nn.ReLU()
            ]
        else:
            model = [
                nn.Conv2DTranspose(input_nz,
                                   ngf * mult,
                                   kernel_size=4,
                                   stride=1,
                                   padding=0,
                                   bias_attr=use_bias),
                norm_layer(ngf * mult),
                nn.ReLU()
            ]

        for i in range(1, n_downsampling):  # add upsampling layers
            mult = 2**(n_downsampling - i)
            output_size = 2**(i+2)
            if norm_type == 'batch':
                model += [
                    nn.Conv2DTranspose(ngf * mult,
                                       ngf * mult//2,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias_attr=use_bias),
                    BatchNorm2D(ngf * mult//2),
                    nn.ReLU()
                ]
            else:
                model += [
                    nn.Conv2DTranspose(ngf * mult,
                                       int(ngf * mult//2),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias_attr=use_bias),
                    norm_layer(int(ngf * mult // 2)),
                    nn.ReLU()
                ]

        output_size = 2**(6)
        model += [
            nn.Conv2DTranspose(ngf,
                               output_nc,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias_attr=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)
