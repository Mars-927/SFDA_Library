import copy
import os

import torch
import torch.optim as optim

from methods.SHOT.shot_utils import *
from model.res_network import feat_classifier, resnet
from utils.Evaluate import test_domain
from utils.Project_Record import Project
def train_source(args, dataset_dirt):
    feature_net = resnet(args.net).cuda()
    classifier_net = feat_classifier(class_num = args.class_num).cuda()

    param_group = [
        {'params': feature_net.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': feature_net.feat_bottleneck.parameters(), 'lr': args.lr},
        {'params': classifier_net.parameters(), 'lr': args.lr}
    ]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    iter_num = 0
    max_iter = args.max_epoch * len(dataset_dirt["train"])
    interval_iter = max_iter // 10

    while iter_num < max_iter:
        # prepare
        feature_net.train()
        classifier_net.train()
        try:
            item = next(iter_source)
            inputs_source, labels_source = item["img"], item["label"]
        except:
            iter_source = iter(dataset_dirt["train"])
            item = next(iter_source)
            inputs_source, labels_source = item["img"], item["label"]

        if inputs_source.size(0) == 1:
            continue
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # network
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = classifier_net(feature_net(inputs_source))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # eval & save
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            feature_net.eval()
            classifier_net.eval()
            acc = test_domain(feature_net, classifier_net, dataset_dirt["test"])
            log_str = 'Task: Train {}, Iter:{}/{}; Accuracy = {:.4f}%'.format(args.source_domain, iter_num, max_iter, acc)
            if acc >= acc_init:
                best_feature_net = copy.deepcopy(feature_net.state_dict())
                best_classifier_net = copy.deepcopy(classifier_net.state_dict())
            Project.log(log_str)

    torch.save(best_feature_net, os.path.join(args.output_dir_src, f"resnet_{args.source_domain}.pt"))
    torch.save(best_classifier_net, os.path.join(args.output_dir_src, f"classifier_{args.source_domain}.pt"))





def shot_src(args, dataset_dirt):
    args.max_epoch = 20
    args.batch_size = 64
    args.lr = 1e-2
    args.net = "resnet50"
    args.bottleneck = 256
    args.epsilon = 1e-5
    args.smooth = 0.1
    args.output_dir_src = Project.root_path
    train_source(args, dataset_dirt)