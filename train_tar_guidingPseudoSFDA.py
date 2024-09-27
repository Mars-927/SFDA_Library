import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *

from model import *
from model.moco_guidingPseudoSFDA import AdaMoCo
from utils import *
from utils.Dataset import get_dataloader_select
from utils.guidingPseudoSFDA_utils import *
from utils.Other import seed_everything
from utils.Project_Record import Project


def prepare_transfer(args, dataloader):
    net = guidingPseudoSFDA(args.num_class)
    momentum_net = guidingPseudoSFDA(args.num_class)
    net.load_weight(args)
    momentum_net.load_weight(args)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)
    moco_model = AdaMoCo(src_model = net, momentum_model = momentum_net, features_length=256, num_classes=args.num_class, dataset_length=len(args.train_dataset_size), temporal_length=args.temporal_length)
    train_loader = dataloader["train"]
    test_loader = dataloader["test"]

    best = 0
    acc, banks, _, _ = eval_and_label_dataset(0, moco_model, None, test_loader)

    for epoch in range(args.num_epochs+1):
        train(epoch, net, moco_model, optimizer, train_loader, ) # train net1 
        acc, banks, gt_labels, pred_labels = eval_and_label_dataset(epoch, moco_model, banks, test_loader)
        if acc > best:
            save_weights(net, epoch, Project.get_folder("root") + '/weights_best.tar')
            best = acc



@torch.no_grad()
def eval_and_label_dataset(epoch, model, banks, test_loader):
    model.eval()
    logits, indices, gt_labels = [], [], []
    features = []
    for batch_idx, batch in enumerate(test_loader):
        inputs, targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3].cuda()
        feats, logits_cls = model(inputs, cls_only=True)
        features.append(feats)
        gt_labels.append(targets)
        logits.append(logits_cls)
        indices.append(idxs)            
    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {"features": features[rand_idxs][: 16384],"probs": probs[rand_idxs][: 16384],"ptr": 0,}

    # refine predicted labels
    pred_labels, _, _, _ = refine_predictions(features, probs, banks, args.num_neighbors) 

    # acc使用accuracy_score函数计算
    acc = 100.*accuracy_score(gt_labels.to('cpu'), pred_labels.to('cpu'))          
    return acc, banks, gt_labels, pred_labels


# Training
def train(epoch, net, moco_model, optimizer, trainloader, banks):
    loss = 0
    acc = 0
    net.train()
    moco_model.train()
    CE = nn.CrossEntropyLoss(reduction='none')
    for batch_idx, batch in enumerate(trainloader): 
        weak_x = batch[0].cuda()
        strong_x = batch[1].cuda()
        y = batch[2].cuda()
        idxs = batch[3].cuda()
        strong_x2 = batch[5].cuda()
        feats_w, logits_w = moco_model(weak_x, cls_only=True)
        if args.label_refinement:
            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, probs_w, banks, args.num_neighbors)
        else:
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w = probs_w.max(1)[1]
        _, logits_q, logits_ctr, keys = moco_model(strong_x, strong_x2)
        if args.ctr:
            loss_ctr = contrastive_loss(logits_ins=logits_ctr,pseudo_labels=moco_model.mem_labels[idxs],mem_labels=moco_model.mem_labels[moco_model.idxs])
        else:
            loss_ctr = 0
        
        # update key features and corresponding pseudo labels
        moco_model.update_memory(epoch, idxs, keys, pseudo_labels_w, y)

        with torch.no_grad():
            #CE weights
            max_entropy = torch.log2(torch.tensor(args.num_class))
            w = entropy(probs_w)
            w = w / max_entropy
            w = torch.exp(-w)
        if args.neg_l:
            loss_cls = ( nl_criterion(logits_q, pseudo_labels_w, args.num_class)).mean()
            if args.reweighting:
                loss_cls = (w * nl_criterion(logits_q, pseudo_labels_w, args.num_class)).mean()
        else:
            loss_cls = ( CE(logits_q, pseudo_labels_w)).mean()
            if args.reweighting:
                loss_cls = (w * CE(logits_q, pseudo_labels_w)).mean()

        loss_div = div(logits_w) + div(logits_q)
        l = loss_cls + loss_ctr + loss_div
        update_labels(banks, idxs, feats_w, logits_w)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += l.item()

        accuracy = 100.*accuracy_score(y.to('cpu'), logits_w.to('cpu').max(1)[1])
        acc += accuracy

    print_log = f'train_loss={loss_cls/len(trainloader):.4f}; train_acc={acc/len(trainloader):.4f}'
    Project.log(print_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default="guidingPseudoSFDA")
    parser.add_argument('--noisy_path', type=str, default=None)
    parser.add_argument('--num_neighbors', default=10, type=int)
    parser.add_argument('--temporal_length', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--ctr', action='store_false', help="use contrastive loss")
    parser.add_argument('--label_refinement', action='store_false', help="Use label refinement")
    parser.add_argument('--neg_l', action='store_false', help="Use negative learning")
    parser.add_argument('--reweighting', action='store_false', help="Use reweighting")
    parser.add_argument('--name', type=str, default="AID_NWPU_UCM_Source")
    parser.add_argument('--method', type=str, default="guidingPseudoSFDA")
    parser.add_argument('--dataset', type=str, default="AID_NWPU_UCM",  choices = ["AID_NWPU_UCM"])
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=str, default='cuda:0')
    parser.add_argument('--weight_basepath', type=str, default="AID_NWPU_UCM_Source")        # 源域模型位置
    args = parser.parse_args()
    device = torch.device("gpu_id")
    seed_everything(args.seed)
    Project(args.name)


    if args.dataset == "AID_NWPU_UCM":
        args.num_class = 10
        args.dataset_path = "dataset/AID_NWPU_UCM"
        args.image_root = "F:/HANS/!dataset/RS_DomainAdaptation_AIDUCMNWPU"
        args.domains = ["AID", "NWPU-RESISC45", "UCMerced_LandUse"]

        for target_domain in args.domains:
            dataset_dirt = get_dataloader_select(args, target_domain)
            for source_domain in args.domains:
                Project.log(f"{args.name} {args.source_domain} => {args.target_domain}")
                args.target_domain = target_domain
                args.source_domain = source_domain
                prepare_transfer(args, dataloader = dataset_dirt)

    elif args.dataset == "OFFICE31":
        args.class_num = 31
        args.dataset_path = r"dataset\OFFICE31"
        args.image_root = "F:/HANS/!dataset/OFFICE31"
        args.domains = ["amazon","dslr","webcam"]
        for target_domain in args.domains:
            dataset_dirt = get_dataloader_select(args, target_domain)
            for source_domain in args.domains:
                Project.log(f"{args.name} {args.source_domain} => {args.target_domain}")
                args.target_domain = target_domain
                args.source_domain = source_domain
                prepare_transfer(args, dataloader = dataset_dirt)
