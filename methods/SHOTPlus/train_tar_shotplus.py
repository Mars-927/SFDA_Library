import os

import numpy as np
import rotation
import torch
import torch.nn as nn
import torch.optim as optim

from methods.SHOTPlus.shotplus_utils import *
from model.res_network import feat_classifier, resnet, rot_feat_classifier
from utils.Evaluate import test_domain
from utils.Project_Record import Project


def train_target_rot(args, dataset_dirt):
    feature_net = resnet(args.net).cuda()
    classifier_net = feat_classifier(class_num = args.class_num).cuda()
    feature_net.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"resnet_{args.source_domain}.pt")))
    feature_net.eval()
    classifier_net.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"classifier_{args.source_domain}.pt")))
    classifier_net.eval()
    netR = rot_feat_classifier(class_num=4, bottleneck_dim=2*args.bottleneck).cuda()

    for k, v in feature_net.named_parameters():
        v.requires_grad = False
    for k, v in classifier_net.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*1}]
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)


    max_iter = args.max_epoch * len(dataset_dirt["train"])
    interval_iter = max_iter // 10
    iter_num = 0
    rot_acc = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            item = next(iter_test)
            inputs_test, tar_idx = item['img'], item['index']
        except:
            iter_test = iter(dataset_dirt["train"])
            item = next(iter_test)
            inputs_test, tar_idx = item['img'], item['index']
        if inputs_test.size(0) == 1:
            continue
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        r_labels_target = np.random.randint(0, 4, len(inputs_test))
        r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
        r_labels_target = torch.from_numpy(r_labels_target).cuda()
        r_inputs_target = r_inputs_target.cuda()
        
        f_outputs = feature_net(inputs_test)
        f_r_outputs = feature_net(r_inputs_target)
        r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

        rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
        rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netR.eval()
            acc_rot = cal_acc_rot(dataset_dirt["train"], feature_net, netR)
            netR.train()

            if rot_acc < acc_rot:
                rot_acc = acc_rot
                best_netR = netR.state_dict()

    return best_netR, rot_acc

def train_target(args, dataset_dirt):
    # prepare network
    feature_net = resnet(args.net).cuda()
    classifier_net = feat_classifier(class_num = args.class_num).cuda()
    feature_net.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"resnet_{args.source_domain}.pt")))
    feature_net.train()
    classifier_net.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"classifier_{args.source_domain}.pt")))
    classifier_net.eval()
    for k, v in classifier_net.named_parameters():
        v.requires_grad = False

    # pretrain rot network
    if not args.ssl == 0:
        netR = rot_feat_classifier(class_num=4, bottleneck_dim=2*args.bottleneck).cuda()
        netR_dict, _ = train_target_rot(args)
        netR.load_state_dict(netR_dict)

    param_group = []
    for k, v in feature_net.backbone.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in feature_net.feat_bottleneck.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    if not args.ssl == 0:
        for k, v in netR.named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dataset_dirt["train"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            item = next(iter_test)
            inputs_test, tar_idx = item['img'], item['index']
        except:
            iter_test = iter(dataset_dirt["train"])
            item = next(iter_test)
            inputs_test, tar_idx = item['img'], item['index']
        if inputs_test.size(0) == 1:
            continue
        inputs_test = inputs_test.cuda()

        
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            feature_net.eval()
            mem_label = obtain_label(dataset_dirt["train"], feature_net, classifier_net, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            feature_net.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        if args.cls_par > 0:
            pred = mem_label[tar_idx]

        features_test = feature_net(inputs_test)
        outputs_test = classifier_net(features_test)

        if args.cls_par > 0:
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred.long())
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        classifier_loss.backward()

        if not args.ssl == 0:
            r_labels_target = np.random.randint(0, 4, len(inputs_test))
            r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()
            f_outputs = feature_net(inputs_test)
            f_outputs = f_outputs.detach()
            f_r_outputs = feature_net(r_inputs_target)
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))
            rotation_loss = args.ssl * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)   
            rotation_loss.backward() 
        optimizer.step()

        # save
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            feature_net.eval()
            acc = test_domain(feature_net, classifier_net, dataset_dirt["test"])
            log_str = 'Task: Train {}, Iter:{}/{}; Accuracy = {:.4f}%'.format(args.target_domain, iter_num, max_iter, acc)
            Project.log(log_str)
            feature_net.train()
            if best < acc:
                best = acc
                torch.save(feature_net.state_dict(), os.path.join(args.output_dir_src, f"resnet_{args.source_domain}2{args.target_domain}.pt"))
                torch.save(classifier_net.state_dict(), os.path.join(args.output_dir_src, f"classifier_{args.source_domain}2{args.target_domain}.pt"))

def shotplus_tar(args, dataset_dirt):
    args.max_epoch = 15
    args.interval = 15
    args.lr = 1e-2
    args.net = "resnet50"
    args.gent = True
    args.ent = True
    args.threshold = 0
    args.cls_par = 0.3
    args.ent_par = 1.0
    args.lr_decay1 = 0.1
    args.lr_decay2 = 1.0
    args.bottleneck = 256
    args.epsilon = 1e-5
    args.distance = "cosine"      # euclidean
    args.ssl = 0.0
    args.output_dir_src = Project.root_path
    train_target(args, dataset_dirt)
    