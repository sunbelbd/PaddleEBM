import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import numpy as np 
from scipy import io as sio

import logging
import torch
import numpy as np 

# currently not used
def heights_initialize(data):
    n, d = data.shape
    heights_tmp = torch.zeros(n,d)
    heights_tmp[:,0] = 1.0

    for i in range(d):
        data_1d = data[:,i]
        data_sorted, idx = torch.sort(data_1d)
        for j in range(1,n):
            heights_tmp[j,i] = heights_tmp[j-1,i] + (data_sorted[j-1]-data_sorted[j])*(j-1)/n;

        heights_tmp[:,i] = heights_tmp[:,i] - torch.mean(heights_tmp[:,i])
        heights_tmp[idx,i] = heights_tmp[:,i]

    return torch.sum(heights_tmp, 1)


def OT_solver(source, target, batch_size, loops=10000, OT_thresh=0.1, heights=None, initialize=False, display=False):
    m, d = source.shape
    n, _ = target.shape
    mu = 1 / m
    nu = 1 / n

    h = torch.zeros(n,1).float().cuda()
    if initialize:
        h = heights_initialize(target).float().cuda()
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

    batch_num = int(m/batch_size)
    if batch_size*batch_num < m:
        batch_num += 1
    E = torch.zeros(loops, 1).cuda()
    area_diff_record_source = torch.zeros(loops, 1).cuda()
    hyperplane_num_record_source = torch.zeros(loops, 1).cuda()

    for i in range(loops):
        value = torch.zeros(m,1).cuda()
        index = torch.zeros(m,1).cuda()
        for j in range(batch_num):
            idx0 = j * batch_size
            idx1 = (j+1) * batch_size
            if idx1 > m:
                idx1 = m 
            source_batch = source[idx0:idx1]
            hyperplanes = target.mm(torch.transpose(source_batch, 0, 1)) + h

            v, ind = torch.max(hyperplanes, 0)
            value[idx0:idx1] = v.view(-1,1)
            index[idx0:idx1] = ind.view(-1,1)

        E[i] = torch.sum(value)/m
        sorted_index, counts = torch.unique(index, return_counts=True)
        sorted_index = sorted_index.type('torch.cuda.LongTensor') 
        counts = counts.type('torch.cuda.FloatTensor')
        delta_h = torch.zeros(n,1).float().cuda()
        delta_h[sorted_index] = counts.view(-1,1) * (1/m) 
        delta_h = delta_h - nu

        area_diff_record_source[i] = torch.sum(torch.abs(delta_h))
        hyperplane_num_record_source[i] = sorted_index.shape[0]

        if i>=1 and i<=50:
            if area_diff_record_source[i] > area_diff_record_source[i-1] and hyperplane_num_record_source[i] < hyperplane_num_record_source[i-1]:
                alpha_source = alpha_source * 1

        if i > 50:
            curr_hyper_num = torch.mean(hyperplane_num_record_source[i-2:i])
            prev_hyper_num = torch.mean(hyperplane_num_record_source[i-5:i-3])
            curr_area_diff = torch.mean(area_diff_record_source[i-2:i])
            prev_area_diff = torch.mean(area_diff_record_source[i-5:i-3])
            curr_energy = torch.mean(E[i-2:i])
            prev_energy = torch.mean(E[i-5:i-3])

            if curr_energy>=prev_energy and curr_area_diff>=prev_area_diff and curr_hyper_num<=prev_hyper_num:
                if alpha_source > 0.00000000001/n**2:
                    alpha_source = alpha_source * 0.995

        mt_source = beta_1 * mt_source + (1-beta_1) * delta_h
        Vt_source = beta_2 * Vt_source + (1-beta_2) * torch.sum(delta_h**2)
        eta_t = alpha_source * mt_source / (torch.sqrt(Vt_source) + epsilon)
        
        eta_t = eta_t - torch.mean(eta_t)
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

    index = torch.zeros(m,1).cuda()
    for j in range(batch_num):
        idx0 = j * batch_size
        idx1 = (j+1) * batch_size
        if idx1 > m:
            idx1 = m 
        source_batch = source[idx0:idx1]
        hyperplanes = target.mm(torch.transpose(source_batch, 0, 1)) + h
        
        _, ind = torch.max(hyperplanes, 0)
        index[idx0:idx1] = ind.view(-1,1)
    index = index.type('torch.cuda.LongTensor')
    index = torch.squeeze(index)


    _, inverse_indices = torch.unique(index, return_inverse=True)
    source_idx = torch.unique(inverse_indices)

    return index, source_idx

def inverse_correspondence_reconstruction(source, target, h, batch_size=5000):
    m, d = source.shape
    n, _ = target.shape
    mu = 1 / m
    nu = 1 / n
    batch_num = int(m/batch_size)
    if batch_size*batch_num < m:
        batch_num += 1

    index = torch.zeros(m,1).cuda()
    for j in range(batch_num):
        idx0 = j * batch_size
        idx1 = (j+1) * batch_size
        if idx1 > m:
            idx1 = m 
        source_batch = source[idx0:idx1]
        hyperplanes = target.mm(torch.transpose(source_batch, 0, 1)) + h
        
        _, ind = torch.max(hyperplanes, 0)
        index[idx0:idx1] = ind.view(-1,1)
    index = index.type('torch.cuda.LongTensor')
    index = torch.squeeze(index)
    sorted_idx, ind = torch.sort(index)

    return target[index]

class decoder(nn.Module):
    def __init__(self, dim_Z=64, dim=64):
        super(decoder, self).__init__()
        def conv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, padding=2, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l0_5 = nn.Sequential(
            conv_bn_relu(3, dim),
            conv_bn_relu(dim, dim * 2),
            conv_bn_relu(dim * 2, dim * 4),
            conv_bn_relu(dim * 4, dim * 8))
        self.l0 = nn.Sequential(
            nn.Linear(dim * 8 * 2 * 2, dim_Z, bias=False))
            
        self.l1 = nn.Sequential(
            nn.Linear(dim_Z, dim * 8 * 2 * 2, bias=False),
            nn.BatchNorm1d(dim * 8 * 2 * 2),
            nn.ReLU())
        self.l1_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
     
    def encode(self, x):
        y = self.l0_5(x)
        y = y.view(y.size(0), -1)
        y = self.l0(y)
        return y

    def decode(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 2, 2)
        y = self.l1_5(y)
        return y

    def forward(self, x):
        y = self.decode(x)
        return 0.5*y+0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warm_steps', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--train_epoch', type=int, default=50)
    parser.add_argument('--ot_batch_size', type=int, default=1000)
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument('--cuda_ids', nargs='+', help='<Required> Set flag', required=True, default=0)
    parser.add_argument('--step_size', type=float, default=3.0)
    parser.add_argument('--random_size', type=float, default=0.01)
    parser.add_argument('--display', type=bool, default=True)
    parser.add_argument('--OT_mode', type=str, default='direct')
    parser.add_argument('--ratio', type=float, default=0.3)
    args = parser.parse_args()

    # training parameters
    batch_size = args.batch_size
    lr = args.lr
    train_epoch = args.train_epoch
    latent_size = args.latent_size
    steps = args.steps
    sigma = args.sigma
    checkpoint = args.checkpoint
    test = args.test
    mode = args.mode
    ot_batch_size = args.ot_batch_size
    inner_steps = args.inner_steps
    cuda_ids = list(map(int, args.cuda_ids)) 
    step_size = args.step_size
    random_size = args.random_size
    warm_steps = args.warm_steps
    display = args.display
    OT_mode = args.OT_mode
    ratio = args.ratio
    torch.cuda.set_device(cuda_ids[0])
    print(cuda_ids)

    # saving folder
    save_path = './SVHN/warm_' + OT_mode + '_' + mode + '_' + str(latent_size) + '_' + str(warm_steps) + '_' + str(steps) + '_' + str(step_size) + '_' + str(inner_steps) + '_' + str(ratio) + '_' + str(random_size) + '_' + str(lr) + '/'
    if not os.path.isdir('./SVHN/'):
        os.mkdir('./SVHN/')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'models/'):
        os.mkdir(save_path+'models/')
    if not os.path.isdir(save_path+'mat/'):
        os.mkdir(save_path+'mat/')
    if not os.path.isdir(save_path+'images/'):
        os.mkdir(save_path+'images/')

    # log file
    logger = logging.getLogger(__name__)  
    logger.setLevel(logging.INFO)
    logging_file = save_path + 'log.log'
    file_handler = logging.FileHandler(logging_file, 'a')
    formatter    = logging.Formatter('%(asctime)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # network
    G = decoder(dim_Z=latent_size)
    if checkpoint>0:
        G.load_state_dict(torch.load(save_path+'models/OT_' + str(checkpoint) + '.pth'))
        Z0 = sio.loadmat(save_path+'mat/Z0.mat')
        Z0 = Z0['Z0']
        Z0 = torch.from_numpy(Z0).float().cuda()

        Z1 = sio.loadmat(save_path+'mat/'+str(checkpoint)+'.mat')
        Z1 = Z1['Z']
        Z1 = torch.from_numpy(Z1).float().cuda()
    G = nn.DataParallel(G, device_ids=cuda_ids).cuda()

    # loss
    MSE_loss = nn.MSELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=[.5, .99])

    # data
    im = sio.loadmat('./data/SVHN.mat')
    im = im['images']
    # data
    if mode == 'part':
        im = im[:1000]

    im = torch.from_numpy(im).float()
    n = im.shape[0]
    batch_num = int(n/batch_size)
    if batch_num*batch_size < n:
        batch_num += 1


    # running parameters
    OT_thresh = 0.05
    s = step_size
    rz = 0.1-0.02*(checkpoint%5)

    # loss for langevin dynamics
    def getloss(pred, x, z, sigma):
        loss = 1/(2*sigma**2) * torch.pow(x - pred, 2).sum() + 1/2 * torch.pow(z, 2).sum()
        loss /= x.size(0)
        return loss

    if not test:
        G.train() 
        if checkpoint==0:    
            Z0 = torch.randn(n,latent_size).float().cuda()
            Z = Z0.clone()
            Z1 = Z0.clone()
            sio.savemat(save_path+'mat/Z0.mat', {"Z0": Z0.cpu().data.numpy()})

        for epoch in range(checkpoint+1, train_epoch):
            if epoch % 5 == 0:
            	rz -= random_size

            # Langevin dynamics
            for j in range(warm_steps):
                idx = torch.randperm(n)
                G_losses = []
                for i in range(batch_num):       
                    idx0 = i * batch_size
                    idx1 = (i+1) * batch_size
                    if idx1 > n:
                        idx1 = n

                    x = im[idx[idx0:idx1]]
                    mini_batch = x.shape[0]
                    x = x.view(-1, 3, 32, 32).float().cuda()
                         
                    z_ = Z1[idx[idx0:idx1]].cpu()
                    z_ = Variable(z_, requires_grad=True).cuda()
                    for k in range(steps):
                        out = G(z_)
                        loss = getloss(out, x, z_, sigma)
                        loss *= s*2/2
                        delta_z = torch.autograd.grad(loss, z_, retain_graph=True)[0]
                        z_.data -= delta_z.data
                        if epoch < train_epoch/1:
                            z_.data += rz*s*torch.randn((mini_batch, latent_size)).cuda()

                    Z1[idx[idx0:idx1]] = z_.data

                    for _ in range(3):
                        x_ = G(z_)
                        loss1 = torch.pow(x - x_, 2).sum() / (x.size(0))
                        G_train_loss = loss1
                        loss2 = G_train_loss
                        G_optimizer.zero_grad()
                        G_train_loss.backward()
                        G_optimizer.step()
                        G_losses.append(loss1.data.item())
                if display:
                    print('epoch: %d, warm step: %d, loss: %5f'% (epoch, j, torch.mean(torch.FloatTensor(G_losses))))
                logger.info('epoch: %d, warm step: %d, loss: %5f', epoch, j, torch.mean(torch.FloatTensor(G_losses)))

            utils.save_image(x_[:64], save_path+'images/warm_'+str(epoch)+'.png', nrow=8)
            z = torch.randn(64, latent_size).cuda()
            x_ = G(z)
            utils.save_image(x_[:64], save_path+'images/warm_'+str(epoch)+'_gen.png', nrow=8)
            torch.save(G.module.state_dict(), save_path+'models/warm_'+str(epoch)+'.pth')
            sio.savemat(save_path+'mat/Z_tmp_'+str(epoch)+'.mat', {"Z": Z1.cpu().data.numpy()})
 
            # optimal transport                   
            if epoch == checkpoint+1:
                h, E, area_diff, hyper_num = OT_solver(Z1, Z0, ot_batch_size, OT_thresh=OT_thresh, loops=20000, display=True)
                index, source_idx = correspondence_reconstruction(Z1, Z0, h)
            if epoch > checkpoint+1:
                h, E, area_diff, hyper_num = OT_solver(Z1, Z0, ot_batch_size, heights=h, loops=20000, OT_thresh=OT_thresh, display=True)
                index, source_idx = correspondence_reconstruction(Z1, Z0, h)

            # reorder
            Z = Z0[index[source_idx]]
            OT_cost = torch.pow(Z1[source_idx]-Z,2).sum()/(Z1.size(0))
            OT_cost = OT_cost.data.cpu()
            Z = Z * ratio + Z1[source_idx]*(1-ratio)
            Z1 = Z0[index]*ratio + Z1*(1-ratio)
            _im = im[source_idx]

            # regression
            for j in range(inner_steps):
                idx = torch.randperm(_im.shape[0])
                G_losses = []
                for i in range(batch_num):
                    idx0 = i * batch_size
                    idx1 = (i+1) * batch_size
                    if idx1 > _im.shape[0]:
                        break
                    if idx1 > n:
                        idx1 = n

                    x = _im[idx[idx0:idx1]]
                    mini_batch = x.shape[0]
                    x = x.view(-1, 3, 32, 32).float().cuda()
                    z_ = Z[idx[idx0:idx1]]

                    x_ = G(z_)
                    loss1 = torch.pow(x - x_, 2).sum() / (x.size(0))
                    G_train_loss = loss1
                    loss2 = G_train_loss
                    G_optimizer.zero_grad()
                    G_train_loss.backward()
                    G_optimizer.step()

                    G_losses.append(G_train_loss.data.item())

                if display:
                    print('epoch: %d, inner step: %d, loss: %5f'% (epoch, j, torch.mean(torch.FloatTensor(G_losses))))
                logger.info('epoch: %d, inner step: %d, loss: %5f', epoch, j, torch.mean(torch.FloatTensor(G_losses)))

            utils.save_image(x_[:64], save_path+'images/'+str(epoch)+'.png', nrow=8)
            z = torch.randn(64, latent_size).cuda()
            x_ = G(z)
            utils.save_image(x_[:64], save_path+'images/'+str(epoch)+'_gen.png', nrow=8)
            if epoch % 1 == 0:
                torch.save(G.module.state_dict(), save_path+'models/OT_'+str(epoch)+'.pth')
                sio.savemat(save_path+'mat/'+str(epoch)+'.mat', {"Z": Z1.cpu().data.numpy()})
            if display:
                print('[%d/%d]: loss_d: %.3f, loss_g: %.3f loss_1: %.3f loss_2: %.3f OT_loss: %.5f, area_diff: %.5f hyper_num: %.1f' % (
                    (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(G_losses)), loss1, loss2, OT_cost, area_diff, hyper_num))

            logger.info('[%d/%d]: loss_d: %.3f, loss_g: %.3f loss_1: %.3f loss_2: %.3f OT_loss: %.5f, area_diff: %.5f hyper_num: %.1f', 
                epoch + 1, train_epoch, torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(G_losses)), loss1, loss2, OT_cost, area_diff, hyper_num)
        sio.savemat(save_path+'mat/Z.mat', {"Z": Z1.cpu().data.numpy()})
    else:
        # test: generate new images
        ims = torch.zeros(10000,3,32,32)
        for i in range(100):
            z = torch.randn(100, latent_size).cuda()
            x = G(z)
            ims[i*100:(i+1)*100] = x.data.cpu()

        sio.savemat('gen_OT.mat', {"images": ims.numpy()})
        utils.save_image(ims[:64], 'gen.png', nrow=8)
