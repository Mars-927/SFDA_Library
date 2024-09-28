import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.Resnet import Classifier, Resnet50
from utils.Dataset import get_dataloader
from utils.Evaluate import test_domain
from utils.Other import seed_everything
from utils.Project_Record import Project

def train_source(args,domain):
    # dataloader
    train_dataloader = get_dataloader(args, domain)

    # network
    resnet = Resnet50().cuda()
    classifier = Classifier(args.class_num).cuda()

    # loss
    CE = nn.CrossEntropyLoss()

    # opt
    optimizer = optim.SGD([
        {'params': resnet.backbone.parameters(),'lr': args.lr}, 
        {'params': resnet.bottleneck_fc.parameters(),'lr': args.lr * 10}, 
        {'params': classifier.parameters(),'lr': args.lr * 10}, ],
        momentum=0.9,weight_decay=5e-4,nesterov=True)
    lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  args.epoch)

    # train
    Project.log(f"\n\n======> Start Train:{domain} <=======")
    for now_epoch in range(args.epoch):
        for data in train_dataloader['train']:
            img = data["img"].cuda()
            label = data["label"].cuda()
            output = classifier(resnet(img))
            loss = CE(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # eval train test
        acc_save = []
        for data in train_dataloader['test']:
            img = data["img"].cuda()
            label = data["label"].cuda()
            output = classifier(resnet(img))
            predict = torch.argmax(torch.softmax(output,dim=1),dim=1)
            tmp = (label == predict).float().cpu().tolist()
            acc_save.extend(tmp)
        acc = np.mean(acc_save)
        log_str = f"[Train {args.dataset} {domain}] Epoch {now_epoch:>3}/{args.epoch:>3}  Acc: {acc:.4f}"
        Project.log(log_str)

    # eval other domain
    for target_domain in args.domains:
        test_dataloader = get_dataloader(args, target_domain)
        acc = test_domain(resnet,classifier,test_dataloader)
        log_str = f"[ Test {args.dataset} {domain}->{target_domain}] Acc: {acc:.4f}"
        Project.log(log_str)

    # save
    torch.save(resnet.state_dict(), Project.get_folder("root") + f'/resnet_{domain}.pt')
    torch.save(classifier.state_dict(), Project.get_folder("root") + f'/classifier_{domain}.pt')
    

def train_src():


    parser = argparse.ArgumentParser(description='SFDA')
    parser.add_argument('--name', type=str, default="Train Source")
    parser.add_argument('--dataset', type=str, default="AID_NWPU_UCM",  choices = ["AID_NWPU_UCM"])
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    Project(args.name)

    if args.dataset == "AID_NWPU_UCM":
        args.class_num = 10
        args.dataset_path = "dataset/AID_NWPU_UCM"
        args.image_root = "F:/HANS/!dataset/RS_DomainAdaptation_AIDUCMNWPU"
        args.domains = ["AID", "NWPU-RESISC45", "UCMerced_LandUse"]
        for source in args.domains:
            train_source(args, source)
    else:
        print("unknown dataset")
