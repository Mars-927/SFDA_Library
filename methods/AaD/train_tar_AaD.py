
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.AaD.AaD_utils import *
from model.res_network import feat_classifier, resnet
from utils import *
from utils.Evaluate import test_domain
from utils.Project_Record import Project


def train_target_decay(args, dataset_dirt):
    netF = resnet(args.net).cuda()
    oldC = feat_classifier(class_num = args.class_num).cuda()
    netF.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"resnet_{args.source_domain}.pt")))
    oldC.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"classifier_{args.source_domain}.pt")))

    optimizer = optim.SGD([
            {"params": netF.backbone.parameters(), "lr": args.lr * 0.1},
            {"params": netF.feat_bottleneck.parameters(), "lr": args.lr * 1},
            {"params": oldC.parameters(), "lr": args.lr * 1},],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    optimizer = op_copy(optimizer)
    acc_init = 0
    loader = dataset_dirt['train']
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, args.bottleneck_dim)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data['img'].cuda()
            indx=data['index']
            output = netF.forward(inputs)
            output_norm = F.normalize(output)
            outputs = oldC(output)
            outputs = nn.Softmax(-1)(outputs)
            if args.sharp:
                outputs = outputs**2 / ((outputs**2).sum(dim=0))
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()
    max_iter = args.max_epoch * len(dataset_dirt['train'])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    oldC.train()
    best = 0
    
    while iter_num < max_iter:
        if iter_num > 0.5 * max_iter:
            args.K = 2
            args.KK = 4

        netF.train()
        oldC.train()

        try:
            item = next(iter_test)
            inputs_test, tar_idx = item['img'], item['index']
        except:
            iter_test = iter(dataset_dirt["train"])
            item = next(iter_test)
            inputs_test, tar_idx = item['img'], item['index']
        if inputs_test.size(0) == 1:
            continue

        if args.alpha_decay:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        inputs_test = inputs_test.cuda()

        iter_num += 1
        if args.lr_decay:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_target = inputs_test.cuda()
        features_test = netF(inputs_target)
        output = oldC(features_test)
        softmax_out = nn.Softmax(dim=1)(output)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()
            pred_bs = softmax_out
            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = pred_bs.detach().clone()
            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C
            fea_near = fea_bank[idx_near]  # batch x K x num_dim

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C

        loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1))

        # other prediction scores as negative pairs
        mask = torch.ones((inputs_target.shape[0], inputs_target.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        if args.noGRAD:
            copy = softmax_out.T.detach().clone()
        else:
            copy = softmax_out.T

        dot_neg = softmax_out @ copy
        dot_neg = (dot_neg * mask.cuda()).sum(-1)
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()
            acc = test_domain(netF, oldC, dataset_dirt["test"])
            log_str = 'Task: Train {}, Iter:{}/{}; Accuracy = {:.4f}%'.format(args.target_domain, iter_num, max_iter, acc)
            Project.log(log_str)
            if best < acc:
                best = acc
                torch.save(netF.state_dict(), os.path.join(args.output_dir_src, f"resnet_{args.source_domain}2{args.target_domain}.pt"))
                torch.save(oldC.state_dict(), os.path.join(args.output_dir_src, f"classifier_{args.source_domain}2{args.target_domain}.pt"))



def AaD_tar(args, dataset_dirt):
    args.max_epoch = 40
    args.interval = 15
    args.lr = 1e-3
    args.bottleneck_dim = 256
    args.net = 'resnet50'
    args.k = 2
    args.K = 4
    args.KK = 3
    args.alpha_decay = False
    args.alpha = 1.0
    args.beta = 0.75
    args.lr_decay = False
    args.noGRAD = False
    args.sharp = False
    args.output_dir_src = Project.root_path
    train_target_decay(args, dataset_dirt)

