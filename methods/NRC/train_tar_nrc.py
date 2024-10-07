import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.NRC.nrc_utils import *
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

    param_group = []
    param_group_c = []
    for k, v in feature_net.backbone.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in feature_net.feat_bottleneck.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in classifier_net.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    #building feature bank and score bank
    loader = dataset_dirt["train"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,args.feature_dim)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    # init feature bank and score bank
    feature_net.eval()
    classifier_net.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data['img'].cuda()
            indx=data['index']
            output = feature_net(inputs)
            output_norm=F.normalize(output)
            outputs = classifier_net(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()
    max_iter = args.max_epoch * len(dataset_dirt["train"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best = 0
    feature_net.train()
    classifier_net.train()

    # train
    while iter_num < max_iter:
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

        # predict
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
        features_test = feature_net(inputs_test)
        outputs_test = classifier_net(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)


        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            # updata bank
            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            # get neighbor feature and predict
            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,dim=-1, largest=True, k=args.K+1)
            idx_near = idx_near[:, 1:]              # batch x K
            score_near = score_bank[idx_near]       # batch x K x C

            # get weight k
            fea_near = fea_bank[idx_near]                                           # batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1)     # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))             # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=args.KK+1)   # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:]                                   # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()                         # batch x K
            weight = torch.where(match > 0., match,torch.ones_like(match).fill_(0.1))   # batch x K

            # get weight kk
            weight_kk = weight.unsqueeze(-1).expand(-1, -1, args.KK)                 # batch x K x M
            weight_kk = weight_kk.fill_(0.1)


            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],-1)  # batch x KM
            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,args.class_num)  # batch x KM x C
            score_self = score_bank[tar_idx]


        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x C x 1
        const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss = torch.mean(const)


        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,-1)  # batch x K x C

        loss += torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *weight.cuda()).sum(1))



        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
        loss += gentropy_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

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


def nrc_tar(args, dataset_dirt):
    args.max_epoch = 15
    args.interval = 15
    args.lr = 1e-3
    args.net = "resnet50"
    args.K = 3
    args.KK = 2
    args.epsilon = 1e-5




    args.feature_dim = 256

    args.output_dir_src = Project.root_path
    train_target(args, dataset_dirt)


