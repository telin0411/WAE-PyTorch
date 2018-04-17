import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class z_adversary(nn.Module):
    def __init__(self, ngpu=1, batch_norm=True, noise='gaussian',
                 nowozin_trick=False, pz_scale=1.0, pz='normal', ifcuda=False):
        super(z_adversary, self).__init__()
        self.ngpu = ngpu
        self.ifcuda = ifcuda
        self.nowozin_trick = nowozin_trick
        self.pz_scale = pz_scale
        if nowozin_trick:
            assert pz == 'normal'
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # Layer 5
            nn.Linear(512, 1),
        )

    def forward(self, input):
        sigma2_p = np.square(self.pz_scale)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hi = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else:
            hi = self.net(input)
        if self.nowozin_trick:
            normsq = torch.sum(input**2, 1, keepdim=False)
            sigma2_p_a = np.asarray(sigma2_p)
            m1 = Variable(0.5 * torch.log(torch.from_numpy(np.asarray(2. * np.pi)))).type(torch.FloatTensor)
            m2 = Variable(0.5 * 64 * torch.log(torch.from_numpy(sigma2_p_a))).type(torch.FloatTensor)
            if self.ifcuda:
                m1, m2 = m1.cuda(), m2.cuda()
                normsq = normsq.cuda()
            hi = hi - m1 - m2 - normsq.view(input.shape[0],1) / 2. / sigma2_p
        return hi


class transform_noise(nn.Module):
    def __init__(self, ngpu=1, batch_norm=True, noise='gaussian'):
        super(transform_noise, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(64, 128),
            nn.Tanh(),
            # Layer 2
            nn.Linear(128, 256),
            nn.Tanh(),
            # Layer 3
            nn.Linear(256, 512),
            nn.Tanh(),
            # Layer 4
            nn.Linear(512, 64**2),
            nn.Tanh(),
        )

    def forward(self, code, eps):
        if isinstance(code.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            A = nn.parallel.data_parallel(self.net, code, range(self.ngpu))
        else:
            A = self.net(code)
        eps = eps.view(-1, 1, 64)
        res = torch.matmul(eps, A)
        res = res.view(-1, 64)
        return res, A
