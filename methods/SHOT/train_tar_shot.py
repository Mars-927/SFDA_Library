import copy
import os

import torch
import torch.optim as optim

from methods.SHOT.shot_utils import *
from model.res_network import feat_classifier, resnet
from utils.Evaluate import test_domain
from utils.Project_Record import Project



def train_target(args, dataset_dirt):
    feature_net = resnet(args.net).cuda()
    classifier_net = feat_classifier(class_num = args.class_num).cuda()
    feature_net.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"resnet_{args.source_domain}.pt")))
    feature_net.train()
    classifier_net.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"classifier_{args.source_domain}.pt")))
    classifier_net.eval()

    for k, v in classifier_net.named_parameters():
        v.requires_grad = False

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
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dataset_dirt["train"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best = 0

    while iter_num < max_iter:
        # prepare 
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
        
        # get pseudo, ever interval iter
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            feature_net.eval()
            mem_label = obtain_label(dataset_dirt["train"], feature_net, classifier_net, args)      # get pseudo label by deep cluster
            mem_label = torch.from_numpy(mem_label).cuda()
            feature_net.train()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # network
        features_test = feature_net(inputs_test)
        outputs_test = classifier_net(features_test)
        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred.long())                  # pseudo ce
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))                                     # ent
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))       # div
                entropy_loss -= gentropy_loss 
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
        optimizer.zero_grad()
        classifier_loss.backward()
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


def shot_tar(args, dataset_dirt):
    # base office31
    # ~/anaconda3/envs/pytorch/bin/python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0
    # ~/anaconda3/envs/pytorch/bin/python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/
    
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
    args.output_dir_src = Project.root_path
    train_target(args, dataset_dirt)