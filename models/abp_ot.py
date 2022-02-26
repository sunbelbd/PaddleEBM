import paddle
import paddle.nn as nn
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel

from .builder import MODELS, get_network

from solver.builder import build_optimizer, build_lr_scheduler
import paddle
import numpy as np 

# currently not used
def heights_initialize(data):
    n, d = data.shape
    heights_tmp = paddle.zeros((n,d))
    heights_tmp[:, 0] = 1.0

    for i in range(d):
        data_1d = data[:, i]
        idx = paddle.argsort(data_1d)
        data_sorted = data_1d[idx]
        for j in range(1, n):
            heights_tmp[j, i] = heights_tmp[j-1, i] + (data_sorted[j-1] - data_sorted[j]) * (j-1) / n

        heights_tmp[:, i] = heights_tmp[:, i] - paddle.mean(heights_tmp[:, i])
        heights_tmp[idx, i] = heights_tmp[:, i]

    return paddle.sum(heights_tmp, 1)

def OT_solver(source, target, batch_size, loops=10000, OT_thresh=0.1, heights=None, initialize=False, display=False):
    m, d = source.shape
    n, _ = target.shape
    mu = 1 / m
    nu = 1 / n

    h = paddle.zeros((n,1))
    if initialize:
        h = heights_initialize(target)
    if heights is not None:
        h = heights

    # params
    thresh = 1e-10
    alpha_source = (n*d)**0.5/100
    beta_1 = 0.9
    beta_2 = 0.5
    epsilon = 1e-16
    mt_source = 0
    Vt_source = 0
    mt_target = 0
    Vt_target = 0

    batch_num = int(m / batch_size)
    if batch_size * batch_num < m:
        batch_num += 1
    E = paddle.zeros((loops, 1))
    area_diff_record_source = paddle.zeros((loops, 1))
    hyperplane_num_record_source = paddle.zeros((loops, 1))

    for i in range(loops):
        value = paddle.zeros((m,1))
        index = paddle.zeros((m,1))
        for j in range(batch_num):
            idx0 = j * batch_size
            idx1 = (j+1) * batch_size
            if idx1 > m:
                idx1 = m 
            source_batch = source[idx0:idx1]
            hyperplanes = target.mm(source_batch.transpose([1, 0])) + h
            
            ind = paddle.argmax(hyperplanes, 0)
            v = paddle.max(hyperplanes, 0)
            value[idx0:idx1] = v.reshape((-1,1))
            index[idx0:idx1] = ind.reshape((-1,1)).cast("float32")

        E[i] = paddle.sum(value) / m
        sorted_index, counts = paddle.unique(index, return_counts=True)
        delta_h = paddle.zeros((n,1))
        delta_h[sorted_index.cast("int")] = counts.reshape((-1,1)) * (1/m) 
        delta_h = delta_h - nu

        area_diff_record_source[i] = paddle.sum(paddle.abs(delta_h))
        hyperplane_num_record_source[i] = sorted_index.shape[0]

        if i>=1 and i<=50:
            if area_diff_record_source[i] > area_diff_record_source[i-1] and hyperplane_num_record_source[i] < hyperplane_num_record_source[i-1]:
                alpha_source = alpha_source * 1

        if i > 50:
            curr_hyper_num = paddle.mean(hyperplane_num_record_source[i-2:i])
            prev_hyper_num = paddle.mean(hyperplane_num_record_source[i-5:i-3])
            curr_area_diff = paddle.mean(area_diff_record_source[i-2:i])
            prev_area_diff = paddle.mean(area_diff_record_source[i-5:i-3])
            curr_energy = paddle.mean(E[i-2:i])
            prev_energy = paddle.mean(E[i-5:i-3])

            if curr_energy>=prev_energy and curr_area_diff>=prev_area_diff and curr_hyper_num<=prev_hyper_num:
                if alpha_source > 0.00000000001/n**2:
                    alpha_source = alpha_source * 0.995

        mt_source = beta_1 * mt_source + (1-beta_1) * delta_h
        Vt_source = beta_2 * Vt_source + (1-beta_2) * paddle.sum(delta_h**2)
        eta_t = alpha_source * mt_source / (paddle.sqrt(Vt_source) + epsilon)
        
        eta_t = eta_t - paddle.mean(eta_t)
        h -= eta_t
        
        # if hyperplane_num_record_source[i] >= 0.99*n or area_diff_record_source[i] <= 0.01:
        if area_diff_record_source[i] <= OT_thresh:
            if display:
                print(i, alpha_source, E[i], area_diff_record_source[i], hyperplane_num_record_source[i])
            break

    return h, E, area_diff_record_source[i], hyperplane_num_record_source[i]

def correspondence_reconstruction(source, target, h, batch_size=5000):
    m, d = source.shape
    n, _ = target.shape
    mu = 1 / m
    nu = 1 / n
    batch_num = int(m/batch_size)
    if batch_size*batch_num < m:
        batch_num += 1

    index = paddle.zeros((m,1))
    for j in range(batch_num):
        idx0 = j * batch_size
        idx1 = (j+1) * batch_size
        if idx1 > m:
            idx1 = m 
        source_batch = source[idx0:idx1]
        hyperplanes = target.mm(source_batch.transpose([0, 1])) + h
        
        ind = paddle.argmax(hyperplanes, 0)
        index[idx0:idx1] = ind.reshape((-1,1)).cast("float32")
    index = index.type('torch.cuda.LongTensor')
    index = paddle.squeeze(index)


    _, inverse_indices = paddle.unique(index, return_inverse=True)
    source_idx = paddle.unique(inverse_indices)

    return index, source_idx

def inverse_correspondence_reconstruction(source, target, h, batch_size=5000):
    m, d = source.shape
    n, _ = target.shape
    mu = 1 / m
    nu = 1 / n
    batch_num = int(m/batch_size)
    if batch_size*batch_num < m:
        batch_num += 1

    index = paddle.zeros((m,1))
    for j in range(batch_num):
        idx0 = j * batch_size
        idx1 = (j+1) * batch_size
        if idx1 > m:
            idx1 = m 
        source_batch = source[idx0:idx1]
        hyperplanes = target.mm(source_batch.transpose([1, 0])) + h
        
        ind = paddle.argmax(hyperplanes, 0)
        index[idx0:idx1] = ind.reshape((-1,1))
    index = paddle.squeeze(index)

    return target[index]

@MODELS.register()
class ABPOT(BaseModel):
    """ This class implements the ABPOT model. <Learning Deep Latent Variable Models by Short-Run MCMC Inference with Optimal Transport Correction>
    
    https://openaccess.thecvf.com/content/CVPR2021/papers/An_Learning_Deep_Latent_Variable_Models_by_Short-Run_MCMC_Inference_With_CVPR_2021_paper.pdf
    
    Paper Author: Dongsheng An, Jianwen Xie, Ping Li
    Author: Yifei Xu
    """

    def __init__(self, generator, ebm, encoder, mcmc, params=None):
        super(ABPOT, self).__init__(params=params)
        self.mcmc_cfg = mcmc
        self.gen_cfg = generator

        self.lr_scheduler = OrderedDict()

        self.input_nz = generator['input_nz']
        # define generator
        self.nets['netG'] = get_network(generator, "Generator")
        self.nets['netEBM'] = get_network(ebm, "EnergyBased")

        encoder.input_sz = ebm.input_sz
        encoder.input_nc = ebm.input_nc
        encoder.output_nc = generator.input_nz
        self.nets['netEnc'] = get_network(encoder, "Encoder")

        self.netG_criterion = nn.loss.MSELoss(reduction='sum')

    def setup_input(self, input):
        self.inputs['obs'] = paddle.to_tensor(input['data'])

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
        self.optimizers['optimVAE'] = build_optimizer(
            cfg.optimVAE, self.lr_scheduler['optimVAE'], self.nets['netG'].parameters() + self.nets['netEnc'].parameters())
        self.optimizers['optimEBM'] = build_optimizer(
            cfg.optimEBM, self.lr_scheduler['optimEBM'], self.nets['netEBM'].parameters())

        return self.optimizers

    def get_z_random(self, batch_size, input_nz):
        random_type = self.gen_cfg.get('random_type', 'normal')
        if random_type == 'normal':
            return paddle.randn(shape=[batch_size, input_nz, 1, 1])
        elif random_type == ' uniform':
            return paddle.rand(shape=[batch_size, input_nz, 1, 1]) * 2.0 - 1.0
        else:
            raise NotImplementedError(
                'Unknown random type: {}'.format(random_type))

    def encode(self, input_image):
        mu, logVar = self.nets['netEnc'](input_image)
        std = paddle.exp(logVar * 0.5)
        eps = self.get_z_random(std.shape[0], std.shape[1])
        z = eps * std + mu
        return z, mu, logVar

    def mcmc_sample(self, init_state):
        cur_state = init_state.detach()
        for i in range(self.mcmc_cfg.num_steps):
            cur_state.stop_gradient = False
            neg_energy = self.nets['netEBM'](cur_state)
            grad = paddle.grad([neg_energy], [cur_state], retain_graph=True)[0]
            noise = paddle.rand(shape=self.inputs['obs'].shape)
            new_state = cur_state - self.mcmc_cfg.step_size * self.mcmc_cfg.step_size * \
                (cur_state / self.mcmc_cfg.refsig / self.mcmc_cfg.refsig -
                 grad) + self.mcmc_cfg.step_size * noise
            cur_state = new_state.detach()
        return cur_state

    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""

        batch_size = self.inputs['obs'].shape[0]
        self.z = self.get_z_random(batch_size, self.input_nz)
        self.fake_gen = self.nets['netG'](self.z)
        self.fake_syn = self.mcmc_sample(self.fake_gen)

        self.syn_z, self.syn_mu, self.syn_logVar = self.encode(self.fake_syn)
        self.ae_res = self.nets['netG'](self.syn_z)

        self.visual_items['real'] = self.inputs['obs']
        self.visual_items['fake_gen'] = self.fake_gen
        self.visual_items['fake_syn'] = self.fake_syn

    def backward_EBM(self):
        self.real_neg_energy = self.nets['netEBM'](self.inputs['obs'])
        self.fake_neg_energy = self.nets['netEBM'](self.fake_syn)

        self.loss_EBM = paddle.sum(self.fake_neg_energy.mean(
            0) - self.real_neg_energy.mean(0))
        self.loss_EBM.backward()
        self.losses['loss_EBM'] = self.loss_EBM

    def backward_VAE(self):
        self.loss_recon = self.netG_criterion(self.fake_gen, self.fake_syn)
        self.loss_kl = paddle.sum(1 + self.syn_logVar - self.syn_mu.pow(
            2) - self.syn_logVar.exp()) * (-0.5 * self.params.lambda_kl)

        self.loss_VAE = self.loss_recon + self.params.lambda_kl * self.loss_kl

        self.loss_VAE.backward()
        self.losses['loss_recon'] = self.loss_recon
        self.losses['loss_kl'] = self.loss_kl

    def train_iter(self, optims=None):
        self.forward()

        # update EBM
        self.set_requires_grad(self.nets['netEBM'], True)
        self.set_requires_grad(self.nets['netG'], False)
        self.set_requires_grad(self.nets['netEnc'], False)
        self.optimizers['optimEBM'].clear_grad()
        self.backward_EBM()
        self.optimizers['optimEBM'].step()

        # update G
        self.set_requires_grad(self.nets['netG'], True)
        self.set_requires_grad(self.nets['netEnc'], True)
        self.set_requires_grad(self.nets['netEBM'], False)
        self.optimizers['optimVAE'].clear_grad()
        self.backward_VAE()
        self.optimizers['optimVAE'].step()
