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
from models.wae_gan_network import Encoder, Decoder
from datasets import data_provider


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

    # Define the discriminator and initialize the weights
    discriminator = z_adversary(ngpu, ifcuda=opt.cuda)
    discriminator.apply(weights_init)

    # define loss functions
    rec_criterion = nn.MSELoss()
    dis_criterion = nn.BCELoss()
    #def d_crit(d_real, d_fake, l, eps=1e-15):
    #    loss = -l * torch.mean(torch.log(d_real + eps) + torch.log(1 - d_fake + eps))
    #    return loss
    #def g_crit(d_fake, l, eps=1e-15):
    #    loss = -l * torch.mean(torch.log(d_fake + eps))
    #    return loss
    def d_crit(d_real, d_fake, l, labels, eps=1e-15):
        sig = nn.Sigmoid()
        if opt.cuda:
            sig = sig.cuda()
        real_labels_ = labels[0]
        fake_labels_ = labels[1]
        loss_Pz = dis_criterion(sig(d_real), real_labels_)
        loss_Qz = dis_criterion(sig(d_fake), fake_labels_)
        loss_adv = l * (loss_Pz + loss_Qz)
        return loss_adv

    def g_crit(d_fake, l, labels, eps=1e-15):
        sig = nn.Sigmoid()
        if opt.cuda:
            sig = sig.cuda()
        real_labels_ = labels[0]
        fake_labels_ = labels[1]
        loss_Gz = dis_criterion(sig(d_fake), real_labels_)
        loss_match = loss_Gz
        return loss_Gz

    # tensor placeholders
    input = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size)
    noise = torch.FloatTensor(opt.batch_size, nz)
    dis_real_label = torch.FloatTensor(opt.batch_size)
    dis_fake_label = torch.FloatTensor(opt.batch_size)
    real_label = 1
    fake_label = 0

    # if using cuda
    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        rec_criterion.cuda()
        discriminator.cuda()
        input = input.cuda()
        dis_real_label, dis_fake_label = dis_real_label.cuda(), dis_fake_label.cuda()
        noise = noise.cuda()

    # define variables
    input = Variable(input)
    noise = Variable(noise)
    dis_real_label = Variable(dis_real_label)
    dis_fake_label = Variable(dis_fake_label)

    # setup optimizer
    optimizerEnc = optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerDec = optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerDis = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizerEnc.load_state_dict(checkpoint['optimizerEnc'])
        optimizerDec.load_state_dict(checkpoint['optimizerDec'])
        optimizerDis.load_state_dict(checkpoint['optimizerDis'])
        print('Starting training at epoch {}'.format(start_epoch))

    avg_loss_R = 0.0
    avg_loss_G = 0.0
    avg_loss_D = 0.0
    # main training loop
    assert start_epoch <= opt.niter
    for epoch in range(start_epoch, opt.niter):
        for i, data in enumerate(dataloader, 0):
            encoder.train()
            decoder.train()
            discriminator.train()

            # zero grads
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()

            # inputs
            real_cpu, label = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.data.resize_as_(real_cpu).copy_(real_cpu)
            dis_real_label.data.resize_(batch_size).fill_(real_label)
            dis_fake_label.data.resize_(batch_size).fill_(fake_label)

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

            encoder.eval()
            sample_noise = Variable(torch.randn(batch_size, 64) * opt.pz_scale)
            if opt.cuda:
                sample_noise = sample_noise.cuda()
            D_real = discriminator(sample_noiese)

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
            D_fake = discriminator(sample_qz)

            dis_labels = (dis_real_label, dis_fake_label)
            loss_d = d_crit(D_real, D_fake, LAMBDA, dis_labels, eps=1e-15)
            loss_d.backward()
            optimizerDis.step()

            encoder.train()
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
            D_fake = discriminator(sample_qz)

            dis_labels = (dis_real_label, dis_fake_label)
            loss_g = g_crit(D_fake, LAMBDA, dis_labels, eps=1e-15)
            loss_g.backward()
            optimizerEnc.step()

            # compute the average loss
            curr_iter = epoch * len(dataloader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_R = avg_loss_R * curr_iter
            all_loss_G += loss_g.data[0]
            all_loss_D += loss_d.data[0]
            all_loss_R += loss_recon.data[0]
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_R = all_loss_R / (curr_iter + 1)

            print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) Reconstruct: %.4f (%.4f)'
                  % (epoch, opt.niter, i, len(dataloader),
                     loss_d.data[0], avg_loss_D, loss_g.data[0], avg_loss_G, loss_recon.data[0], avg_loss_R))
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
            'discriminator': discriminator.state_dict(),
            'best_mse': best_mse,
            'optimizerEnc' : optimizerEnc.state_dict(),
            'optimizerDec' : optimizerDec.state_dict(),
            'optimizerDis' : optimizerDis.state_dict(),
        }, is_best, path=opt.outf)
