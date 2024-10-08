import argparse
import os, sys
sys.path.append('./')

import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
import random, pdb, math, copy
import pickle
from utils import *
from torch import autograd
from utils.Project_Record import Project
from methods.GSFDA.gsfda_utils import *
from model.res_gsfda import feat_classifier, resnet
import torch.nn.functional as F

def train_target_near(args, dataset_dirt):
    netF = resnet(args.net).cuda()
    oldC = feat_classifier(class_num = args.class_num).cuda()
    netF.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"resnet_{args.source_domain}.pt")))
    oldC.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"classifier_{args.source_domain}.pt")))


    param_group_bn = []
    for k, v in netF.backbone.named_parameters():
        if k.find('bn') != -1:
            param_group_bn += [{'params': v, 'lr': args.lr}]
    optimizer = optim.SGD([
        {'params': netF.feat_bottleneck.parameters(),'lr': args.lr * 10}, 
        {'params': oldC.parameters(),'lr': args.lr * 10}] + param_group_bn,
        momentum=0.9,weight_decay=5e-4,nesterov=True)
    optimizer = op_copy(optimizer)
    smax = 100

    loader = dataset_dirt["train"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    netF.eval()
    oldC.eval()
    best = 0
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data['img']
            indx = data['index']
            inputs = inputs.cuda()
            output, _ = netF.forward(inputs, t=1)  
            output_norm = F.normalize(output)
            outputs = oldC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()  
    max_iter = args.max_epoch * len(loader)
    interval_iter = max_iter // args.interval
    iter_num = 0
    netF.train()
    oldC.train()

    while iter_num < max_iter:
        netF.train()
        oldC.train()
        iter_target = iter(dataset_dirt["train"])
        try:
            item = next(iter_target)
            inputs_test, indx = item['img'], item['index']
        except:
            iter_target = iter(dataset_dirt["train"])
            item = next(iter_target)
            inputs_test, indx = item['img'], item['index']
        if inputs_test.size(0) == 1:
            continue
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter) # learning rate decay
        inputs_target = inputs_test.cuda()
        output_f, masks = netF(inputs_target, t=1, s=smax)
        masks_old = masks
        output = oldC(output_f)
        softmax_out = nn.Softmax(dim=1)(output)
        output_re = softmax_out.unsqueeze(1) 

        with torch.no_grad():
            fea_bank[indx].fill_(-0.1)  #do not use the current mini-batch in fea_bank
            #fea_bank=fea_bank.numpy()
            output_f_ = F.normalize(output_f).cpu().detach().clone()
            distance = output_f_ @ fea_bank.t()
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=2)
            score_near = score_bank[idx_near]  
            score_near = score_near.permute(0, 2, 1)
            fea_bank[indx] = output_f_.detach().clone().cpu()
            score_bank[indx] = softmax_out.detach().clone()  #.cpu()

        const = torch.log(torch.bmm(output_re, score_near)).sum(-1)
        loss_const = -torch.mean(const)
        msoftmax = softmax_out.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = im_div + loss_const 
        optimizer.zero_grad()
        loss.backward()
        
        for n, p in netF.feat_bottleneck.named_parameters():
            if n.find('bias') == -1:
                mask_ = ((1 - masks_old)).view(-1, 1).expand(256, 2048).cuda()
                p.grad.data *= mask_
            else:  #no bias here
                mask_ = ((1 - masks_old)).squeeze().cuda()
                p.grad.data *= mask_

        for n, p in oldC.named_parameters():
            if n.find('weight_v') != -1:
                masks__ = masks_old.view(1, -1).expand(args.class_num, 256)
                mask_ = ((1 - masks__)).cuda()
                p.grad.data *= mask_

        for n, p in netF.bn.named_parameters():
            mask_ = ((1 - masks_old)).view(-1).cuda()
            p.grad.data *= mask_

        torch.nn.utils.clip_grad_norm(netF.parameters(), 10000)
        optimizer.step()



        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()
            acc1, _ = cal_acc_sda(dataset_dirt['test'], netF, oldC, t=1)  #1
            log_str = 'Iter:{}/{}; Accuracy on target = {:.2f}%.%'.format(iter_num, max_iter, acc1 * 100)
            Project.log(log_str)
            if best < acc1:
                best = acc1
                torch.save(netF.state_dict(), os.path.join(args.output_dir_src, f"resnet_{args.source_domain}2{args.target_domain}.pt"))
                torch.save(oldC.state_dict(), os.path.join(args.output_dir_src, f"classifier_{args.source_domain}2{args.target_domain}.pt"))




    


def gsfda_tar(args, dataset_dirt):
    args.max_epoch = 32
    args.interval = 15
    args.lr = 1e-3
    args.net = "resnet50"
    args.bottleneck = 256
    args.output_dir_src = Project.root_path
    train_target_near(args)
    
