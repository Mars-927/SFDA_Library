import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from methods.GSFDA.gsfda_utils import *
from model.res_gsfda import feat_classifier, resnet
from utils import *
from utils.Project_Record import Project


def train_source(args, dataset_dirt):
    feature_net = resnet(args.net).cuda()
    classifier_net = feat_classifier(class_num = args.class_num).cuda()
    param_group = [
        {'params': feature_net.backbone.parameters(), 'lr': args.lr},
        {'params': feature_net.feat_bottleneck.parameters(), 'lr': args.lr * 10},
        {'params': feature_net.em.parameters(), 'lr': args.lr * 10},
        {'params': classifier_net.parameters(), 'lr': args.lr * 10}
    ]
    optimizer = optim.SGD(param_group,momentum=0.9,weight_decay=5e-4,nesterov=True)
    smax = 100
    acc_init = 0

    for epoch in range(args.max_epoch):
        feature_net.train()
        classifier_net.train()
        iter_source = iter(dataset_dirt["train"])
        for batch_idx, item in enumerate(iter_source):
            inputs_source, labels_source = item["img"].cuda(), item["label"].cuda()
            if inputs_source.size(0) == 1:
                continue
            progress_ratio = batch_idx / (len(dataset_dirt["train"]) - 1)
            s = (smax - 1 / smax) * progress_ratio + 1 / smax
            outputs, masks = feature_net(inputs_source, 0, s, True)
            output0 = classifier_net(outputs[0])
            output1 = classifier_net(outputs[1])
            reg = 0
            count = 0
            for m in masks[0]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
            for m in masks[1]:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()
            reg /= count
            loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(
                output0, labels_source) + CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(
                    output1, labels_source) + 0.15 * reg

            optimizer.zero_grad()
            loss.backward()

            # Compensate embedding gradients
            for n, p in feature_net.em.named_parameters():
                num = torch.cosh(torch.clamp(s * p.data, -10, 10)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= smax / s * num / den
            torch.nn.utils.clip_grad_norm(feature_net.parameters(), 10000)
            optimizer.step()


        feature_net.eval()
        classifier_net.eval()

        acc_s_tr1, _ = cal_acc_sda(dataset_dirt["test"], feature_net, classifier_net)
        acc_s_tr2, _ = cal_acc_sda(dataset_dirt["test"], feature_net, classifier_net, t=1)

        log_str = 'Iter:{}/{}; Accuracy = {:.2f}%({:.2f}%)'.format(epoch + 1, args.max_epoch, acc_s_tr1 * 100, acc_s_tr2 * 100)
        Project.log(log_str)


        if acc_s_tr1 >= acc_init:
            acc_init = acc_s_tr1
            best_netF = feature_net.state_dict()
            best_netC = classifier_net.state_dict()
    torch.save(best_netF, os.path.join(args.output_dir_src, f"resnet_{args.source_domain}.pt"))
    torch.save(best_netC, os.path.join(args.output_dir_src, f"classifier_{args.source_domain}.pt"))




def gsfda_src(args, dataset_dirt):
    # base office home
    args.max_epoch = 20
    args.batch_size = 64
    args.lr = 1e-3
    args.net = "resnet50"
    args.bottleneck = 256
    args.smooth = 0.1
    args.output_dir_src = Project.root_path
    train_source(args, dataset_dirt)
