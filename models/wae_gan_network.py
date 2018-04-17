import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, ngpu=1, batch_norm=True, noise='gaussian', pz='sphere'):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.noise = noise
        self.pz = pz
        if batch_norm:
            self.net = nn.Sequential(
                # Layer 1
                nn.Conv2d(3, 128, 5, 2, 0, bias=False),
                nn.BatchNorm2d(128, momentum=0.9),
                nn.ReLU(inplace=True),
                # Layer 2
                nn.Conv2d(128, 256, 5, 2, 0, bias=False),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.ReLU(inplace=True),
                # Layer 3
                nn.Conv2d(256, 512, 5, 2, 0, bias=False),
                nn.BatchNorm2d(512, momentum=0.9),
                nn.ReLU(inplace=True),
                # Layer 4
                nn.Conv2d(512, 1024, 5, 2, 0, bias=False),
                nn.BatchNorm2d(1024, momentum=0.9),
                nn.ReLU(inplace=True),
            )
        else:
            self.net = nn.Sequential(
                # Layer 1
                nn.Conv2d(3, 128, 5, 2, 0, bias=False),
                nn.ReLU(inplace=True),
                # Layer 2
                nn.Conv2d(128, 256, 5, 2, 0, bias=False),
                nn.ReLU(inplace=True),
                # Layer 3
                nn.Conv2d(256, 512, 5, 2, 0, bias=False),
                nn.ReLU(inplace=True),
                # Layer 4
                nn.Conv2d(512, 1024, 5, 2, 0, bias=False),
                nn.ReLU(inplace=True),
            )
        if noise != 'gaussian':
            self.L = nn.Sequential(
                nn.Linear(1024, 64),
            )
        else:
            self.mean = nn.Sequential(
                nn.Linear(1024, 64),
            )
            self.log_sigmas = nn.Sequential(
                nn.Linear(1024, 64),
            )

        self.tanh = nn.Tanh()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
            flat = conv.view(-1, 1*1*1024)
            if self.noise != 'gaussian':
                res = nn.parallel.data_parallel(self.L, flat, range(self.ngpu))
                if self.pz == 'uniform':
                    res = self.tanh(res)
            else:
                mean = nn.parallel.data_parallel(self.mean, flat, range(self.ngpu))
                log_sigmas = nn.parallel.data_parallel(self.log_sigmas, flat, range(self.ngpu))
                if self.pz == 'uniform':
                    mean, log_sigmas = self.tanh(mean), self.tanh(log_sigmas)
                return mean, log_sigmas
        else:
            conv = self.net(input)
            flat = conv.view(-1, 1*1*1024)
            if self.noise != 'gaussian':
                res = self.L(flat)
                if self.pz == 'uniform':
                    res = self.tanh(res)
            else:
                mean = self.mean(flat)
                log_sigmas = self.log_sigmas(flat)
                if self.pz == 'uniform':
                    mean, log_sigmas = self.tanh(mean), self.tanh(log_sigmas)
                return mean, log_sigmas
        return res, None


class Decoder(nn.Module):
    def __init__(self, ngpu, batch_norm=True, input_normalize_sym=False):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.input_normalize_sym = input_normalize_sym
        self.L = nn.Sequential(
            nn.Linear(64, 1024*4*4),
            nn.ReLU(inplace=True),
        )

        if batch_norm:
            self.net = nn.Sequential(
                # Layer 1
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512, momentum=0.9),
                nn.ReLU(inplace=True),
                # Layer 2
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.ReLU(inplace=True),
                # Layer 3
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128, momentum=0.9),
                nn.ReLU(inplace=True),
                # Layer 4
                nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            )
        else:
            self.net = nn.Sequential(
                # Layer 1
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True),
                # Layer 2
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True),
                # Layer 3
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True),
                # Layer 4
                nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise):
        if isinstance(noise.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            fc = nn.parallel.data_parallel(self.L, noise, range(self.ngpu))
            fc = fc.view(-1, 1024, 4, 4)
            last_h = nn.parallel.data_parallel(self.net, fc, range(self.ngpu))
        else:
            fc = self.L(noise)
            fc = fc.view(-1, 1024, 4, 4)
            last_h = self.net(fc)
        if self.input_normalize_sym:
            return self.tanh(last_h), last_h
        else:
            return self.sigmoid(last_h), last_h
