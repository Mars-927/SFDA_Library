import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import torch
import torch.nn as nn
import os
from model.Resnet import Resnet50,Classifier
from sklearn.metrics import accuracy_score

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors):
    pred_probs = []
    pred_probs_all = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)
    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)
    _, pred_labels_all = pred_probs_all.max(dim=2)
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)
    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def refine_predictions(features,probs,banks, num_neighbors):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(features, feature_bank, probs_bank, num_neighbors)
    return pred_labels, probs, pred_labels_all, pred_labels_hard


def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()
    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) 
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    loss = F.cross_entropy(logits_ins, labels_ins)
    return loss

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)
    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))
    return loss_div

def nl_criterion(output, y, num_class):
    output = torch.log( torch.clamp(1.-F.softmax(output, dim=1), min=1e-5, max=1.) )
    labels_neg = ( (y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, num_class).cuda()) % num_class ).view(-1)
    l = F.nll_loss(output, labels_neg, reduction='none')
    return l


class guidingPseudoSFDA(nn.Module):
    def __init__(self, num_class):
        super(guidingPseudoSFDA, self).__init__()
        self.resnet = Resnet50()
        self.classifier = Classifier(num_class)
        
    def forward(self, x):
        features = self.resnet(x)
        logits = self.classifier(features)
        return features, logits
    
    def load_weight(self,args):
        self.resnet.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"resnet_{args.source_domain}.pt")))
        self.classifier.load_state_dict(torch.load(os.path.join(args.weight_basepath, f"classifier_{args.source_domain}.pt")))

        

def save_weights(model, e, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save({'epochs': e,'weights': model.state_dict()}, filename)


@torch.no_grad()
def eval_and_label_dataset(model, test_loader, num_neighbors):
    model.eval()
    logits, indices, gt_labels = [], [], []
    features = []
    for _, batch in enumerate(test_loader):

        inputs = batch["weak_augmented"].cuda()
        targets = batch["imgs_label"].cuda()
        idxs = batch["index"].cuda()

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
    pred_labels, _, _, _ = refine_predictions(features, probs, banks, num_neighbors) 

    # acc使用accuracy_score函数计算
    acc = 100.0 * accuracy_score(gt_labels.to('cpu'), pred_labels.to('cpu'))          
    return acc, banks, gt_labels, pred_labels