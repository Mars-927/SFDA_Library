import numpy as np
import torch


@torch.no_grad()
def test_domain(resnet,classifier,test_dataloader):
    # eval
    resnet.eval()
    classifier.eval()

    # test
    acc_save = []
    for data in test_dataloader["all"]:
        img = data["img"].cuda()
        label = data["label"].cuda()
        output = classifier(resnet(img))
        predict = torch.argmax(torch.softmax(output,dim=1),dim=1)
        tmp = (label == predict).float().cpu().tolist()
        acc_save.extend(tmp)
    acc = np.mean(acc_save)

    # train
    resnet.train()
    classifier.train()
    return acc
