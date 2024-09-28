import numpy as np
import torch
from model.res_network import feat_classifier, resnet
import os


@torch.no_grad()
def test_domain(extractor, classifier, test_dataloader):
    # test
    acc_save = []
    for data in test_dataloader:
        img = data["img"].cuda()
        label = data["label"].cuda()
        output = classifier(extractor(img))
        predict = torch.argmax(torch.softmax(output,dim=1),dim=1)
        tmp = (label == predict).float().cpu().tolist()
        acc_save.extend(tmp)
    acc = np.mean(acc_save)
    return acc

@torch.no_grad()
def test_target_shot(args, test_dataloader, weight_basepath):
    feature_net = resnet(args.net).cuda().eval()
    classifier_net = feat_classifier(class_num = args.class_num).cuda().eval()
    feature_net.load_state_dict(torch.load(os.path.join(weight_basepath, f"resnet_{args.source_domain}.pt")))
    classifier_net.load_state_dict(torch.load(os.path.join(weight_basepath, f"classifier_{args.source_domain}.pt")))
    return test_domain(feature_net, classifier_net, test_dataloader)

