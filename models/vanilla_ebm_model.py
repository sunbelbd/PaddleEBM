import paddle
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel

from .builder import MODELS, get_network, get_sampling

from solver.builder import build_optimizer, build_lr_scheduler


@MODELS.register()
class VanillaEBM(BaseModel):
    """ This class implements the vanilla EBM model.
        Support: Image, Point Cloud. 
        TODO: support text generation, trajectory prediction, 3D voxel etc. 
    """

    def __init__(self, ebm, mcmc, generator=None, params=None):

        super(VanillaEBM, self).__init__(params=params)
        self.nets['netEBM'] = get_network(ebm, "EnergyBased")
        self.nets['netG'] = get_network(generator, "Generator")
        mcmc.net = self.nets['netEBM']
        self.sampling = get_sampling(mcmc)
        self.mcmc_config = mcmc
        self.lr_scheduler = OrderedDict()
        self.ws_buffer = {} if getattr(self.params, "warm_start", False) else None
        self.add_noise_to_obs = getattr(self.params, "add_noise_to_obs", 0)
        self.num_chain = getattr(self.params, "num_chain", 1)


    def setup_input(self, input):
        self.inputs['obs'] = paddle.to_tensor(input['data'].astype(np.float32))
        if self.add_noise_to_obs > 0:
            self.inputs['obs'] += self.add_noise_to_obs * paddle.randn(shape=self.inputs['obs'].shape)

        # warm start initialization
        if self.ws_buffer is not None: 
            assert 'index' in input
            fake_init = [] 
            for ii in input['index']:
                i = int(ii)
                if i in self.ws_buffer:
                    fake_init.append(self.ws_buffer[i])
                else: 
                    shapes = self.inputs['obs'].shape
                    shapes[0] = self.num_chain
                    if self.ws_buffer == "Uniform":
                        temp = paddle.rand(shape=shapes) * 2 - 1
                    else: 
                        temp = paddle.rand(shape=shapes) * 2 - 1 if self.mcmc_config.refsig == 0 \
                        else paddle.randn(shape=shapes) * self.mcmc_config.refsig
                    fake_init.append(temp)
                    self.ws_buffer[i] = temp 
            self.inputs['idx'] = input['index']
            self.inputs['fake_init'] = paddle.to_tensor(paddle.concat(fake_init))

    def setup_optimizers(self, cfg):
        iters_per_epoch = cfg.pop('iters_per_epoch')
        for optim in cfg:
            opt_cfg = cfg[optim].copy()
            lr = opt_cfg.pop('learning_rate')
            if 'lr_scheduler' in opt_cfg:
                lr_cfg = opt_cfg.pop('lr_scheduler')
                lr_cfg['learning_rate'] = lr
                lr_cfg['iters_per_epoch'] = iters_per_epoch
                self.lr_scheduler[optim] = build_lr_scheduler(lr_cfg)
            else:
                self.lr_scheduler[optim] = lr
            cfg[optim] = opt_cfg
        if self.nets['netG'] is not None:
            self.optimizers['optimG'] = build_optimizer(
                cfg.optimG, self.lr_scheduler['optimG'], self.nets['netG'].parameters())
        self.optimizers['optimEBM'] = build_optimizer(
            cfg.optimEBM, self.lr_scheduler['optimEBM'], self.nets['netEBM'].parameters())

        return self.optimizers

    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""

        batch_size = self.inputs['obs'].shape[0]

        if self.nets['netG'] is None: 
            if self.ws_buffer is None:
                self.fake_syn = self.sampling(shape=self.inputs['obs'].shape)
            else: 
                self.fake_syn = self.sampling(init=self.inputs['fake_init'])
        else: 
            self.z = paddle.randn(shape=(batch_size, self.input_nz, 1, 1))
            self.fake_gen = self.nets['netG'](self.z)
            self.visual_items['fake_gen'] = self.fake_gen
            self.fake_syn = self.sampling(self.fake_gen)
        if self.ws_buffer is not None:
            for i, id in enumerate(self.inputs['idx']):
                self.ws_buffer[int(id)] = self.fake_syn[i*self.num_chain:(i+1)*self.num_chain]

        self.visual_items['real'] = self.inputs['obs']
        self.visual_items['fake_syn'] = self.fake_syn
        # print("syn_stat: ", float(self.fake_syn.mean()), float(self.fake_syn.std()))

    def backward_EBM(self):
        self.real_neg_energy = self.nets['netEBM'](self.inputs['obs'])
        self.fake_neg_energy = self.nets['netEBM'](self.fake_syn)
        # print(float(self.real_neg_energy.mean()), float(self.fake_neg_energy.mean()))
        # assert False
        self.loss_EBM = self.fake_neg_energy.mean() - self.real_neg_energy.mean()
        self.loss_EBM.backward()
        self.losses['loss_EBM'] = self.loss_EBM

    def backward_G(self):
        self.loss_G = self.netG_criterion(self.fake_gen, self.fake_syn)
        self.loss_G.backward()
        self.losses['loss_G'] = self.loss_G

    def train_iter(self, optims=None):
        self.forward()

        # update EBM
        self.set_requires_grad(self.nets['netEBM'], True)
        if self.nets['netG'] is not None:
            self.set_requires_grad(self.nets['netG'], False)
        self.optimizers['optimEBM'].clear_grad()
        self.backward_EBM()
        self.optimizers['optimEBM'].step()

        # update G
        if self.nets['netG'] is not None:
            self.set_requires_grad(self.nets['netG'], True)
            self.set_requires_grad(self.nets['netEBM'], False)
            self.optimizers['optimG'].clear_grad()
            self.backward_G()
            self.optimizers['optimG'].step()
