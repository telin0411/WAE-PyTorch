"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc, save_checkpoint
from models.common import z_adversary, transform_noise
from models.wae_mmd_network import Encoder, Decoder
from datasets import data_provider


def kernel(opt, sample_qz, sample_pz):
    # compute distances for pz
    norms_pz = torch.sum(sample_pz**2, dim=1, keepdim=True)
    dotprods_pz = torch.matmul(sample_pz, torch.transpose(sample_pz, 0, 1))
    dist_pz = norms_pz + torch.transpose(norms_pz, 0, 1) - 2. * dotprods_pz
    # compute distances for qz
    norms_qz = torch.sum(sample_qz**2, dim=1, keepdim=True)
    dotprods_qz = torch.matmul(sample_qz, torch.transpose(sample_qz, 0, 1))
    dist_qz = norms_qz + torch.transpose(norms_qz, 0, 1) - 2. * dotprods_qz
    # compute overall distances
    dotprods = torch.matmul(sample_qz, torch.transpose(sample_pz, 0, 1))
    dist = norms_qz + torch.transpose(norms_pz, 0, 1) - 2. * dotprods
    # params
    batch_size = sample_qz.size()[0]
    n = batch_size
    nf = float(n)
    half_size = int((n * n - n) / 2)
    # eye matrix
    eye = torch.eye(n)
    eye_var = Variable(eye)
    if opt.cuda:
        eye_var = eye_var.cuda()

    mode = opt.kernel
    if mode == 'IMQ':
        # prelims
        Cbase = 2. * 64 * (opt.pz_scale**2)
        stat = 0.
        # scales
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + dist_qz)
            res1 += C / (C + dist_pz)
            res1 = torch.mul(res1, 1. - eye_var)
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = C / (C + dist)
            res2 = torch.sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    elif mode == 'RBF':
        sigma2_k = torch.topk(dist.view(-1), half_size)[0][half_size-1]
        sigma2_k += torch.topk(dist_qz.view(-1), half_size)[0][half_size-1]
        res1 = torch.exp( -dist_qz / 2. / sigma2_k) + torch.exp( -dist_pz / 2. / sigma2_k)
        res1 = torch.mul(res1, 1. - eye_var)
        res1 = torch.sum(res1) / (nf**2 - nf)
        res2 = torch.exp( -dist / 2. / sigma2_k)
        res2 = torch.sum(res2) * 2. / (nf**2)
        stat = res1 - res2
    else:
        raise NotImplementedError
    return stat


def train(opt):
    # dataset
    dataloader = data_provider(opt.dataroot, opt.batch_size, isCrop=True, mode=opt.dataset)

    # some hyper parameters
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3
    LAMBDA = opt.LAMBDA

    # Define the encoder and initialize the weights
    encoder = Encoder(ngpu, noise=opt.noise)
    encoder.apply(weights_init)

    # Define the decoder and initialize the weights
    decoder = Decoder(ngpu)
    decoder.apply(weights_init)

    # define loss functions
    rec_criterion = nn.MSELoss()
    dis_criterion = nn.BCELoss()

    # tensor placeholders
    input = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size)
    noise = torch.FloatTensor(opt.batch_size, nz)

    # if using cuda
    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        rec_criterion.cuda()
        input = input.cuda()
        noise = noise.cuda()

    # define variables
    input = Variable(input)
    noise = Variable(noise)

    # setup optimizer
    optimizerEnc = optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerDec = optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # loading the pre-trained weights
    start_epoch = 0
    total_iter = 0
    best_mse = 1e10
    if opt.checkpoint != '':
        print("=> loading checkpoint '{}'".format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        start_epoch = checkpoint['epoch']
        total_iter = checkpoint['total_iter']
        best_mse = checkpoint['best_mse']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizerEnc.load_state_dict(checkpoint['optimizerEnc'])
        optimizerDec.load_state_dict(checkpoint['optimizerDec'])
        print('Starting training at epoch {}'.format(start_epoch))

    avg_loss_R = 0.0
    avg_loss_Z = 0.0
    # main training loop
    assert start_epoch <= opt.niter
    for epoch in range(start_epoch, opt.niter):
        for i, data in enumerate(dataloader, 0):
            encoder.train()
            decoder.train()

            # zero grads
            encoder.zero_grad()
            decoder.zero_grad()

            # inputs
            real_cpu, label = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.data.resize_as_(real_cpu).copy_(real_cpu)

            z_mean, z_sigmas = encoder(input)
            if opt.noise == "gaussain":
                z_sigmas = torch.clamp(z_sigmas, -50, 50)
                noise_real_add = torch.randn(batch_size, 64)
                noise_real_add = Variable(noise_real_add)
                if opt.cuda:
                    noise_real_add = noise_real_add.cuda()
                z_encoded = z_mean + torch.mul(noise_real_add, torch.sqrt(1e-8 + torch.exp(z_sigmas)))
            else:
                z_encoded = z_mean

            input_rec, _ = decoder(z_encoded)
            loss_recon = rec_criterion(input_rec, input)
            loss_recon.backward()
            optimizerDec.step()
            optimizerEnc.step()

            # mmd penalty
            sample_noise = Variable(torch.randn(batch_size, 64) * opt.pz_scale)
            if opt.cuda:                                                           
                sample_noise = sample_noise.cuda()                               
                                                                                   
            sample_qz_mean, sample_qz_sigmas = encoder(input)                      
            if opt.noise == "gaussain":                                            
                sample_qz_sigmas = torch.clamp(sample_qz_sigmas, -50, 50)          
                noise_fake_add = torch.randn(batch_size, 64)                       
                noise_fake_add = Variable(noise_fake_add)                          
                if opt.cuda:                                                       
                    noise_fake_add = noise_fake_add.cuda()                      
                sample_qz = sample_qz_mean + torch.mul(noise_fake_add, torch.sqrt(1e-8 + torch.exp(sample_qz_sigmas)))
            else:                                                               
                sample_qz = sample_qz_mean                                      
            loss_z = kernel(opt, sample_qz, sample_noise)
            loss_z.backward()
            optimizerEnc.step()

            # compute the average loss
            curr_iter = epoch * len(dataloader) + i
            all_loss_Z = avg_loss_Z * curr_iter
            all_loss_R = avg_loss_R * curr_iter
            all_loss_Z += loss_z.data[0]
            all_loss_R += loss_recon.data[0]
            avg_loss_Z = all_loss_Z / (curr_iter + 1)
            avg_loss_R = all_loss_R / (curr_iter + 1)

            print('[%d/%d][%d/%d] Loss_Z: %.4f (%.4f) Reconstruct: %.4f (%.4f)'
                  % (epoch, opt.niter, i, len(dataloader),
                     loss_z.data[0], avg_loss_Z, loss_recon.data[0], avg_loss_R))
            if i % 100 == 0:
                decoder.eval()
                vutils.save_image(
                    real_cpu, '%s/real_samples.png' % opt.outf)
                vutils.save_image(
                    input_rec.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
                )
                print ("saved output images to {}".format(opt.outf))

        # do checkpointing
        is_best = loss_recon.data[0] < best_mse
        if is_best:
            best_mse = loss_recon
        save_checkpoint({
            'epoch': epoch + 1,
            'total_iter': total_iter,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'best_mse': best_mse,
            'optimizerEnc' : optimizerEnc.state_dict(),
            'optimizerDec' : optimizerDec.state_dict(),
        }, is_best, path=opt.outf)
