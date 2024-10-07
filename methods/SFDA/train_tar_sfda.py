import os

import torch
import torch.optim as optim
from torchvision.transforms import v2


# from model.Attention import Embedding_Attention
from model.CTRR_model import SimSiam
from model.res_network import resnet, feat_classifier


from utils.Logic_Storage import Logic_Save
from utils.Losses import *

from methods.SFDA.sfda_utils import ema_update_CTRR
from utils.Output_Info import *
from utils.Project_Record import Project
from utils.Select import get_confidence_change
from utils.Evaluate import test_domain


def target_transfer(args, target_dataloader):

    # model
    model = SimSiam(args).cuda()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  args.epoch)

    # teacher model
    resnet_teacher = resnet().cuda()
    resnet_teacher.load_state_dict(torch.load(os.path.join(args.weight_basepath,f"resnet_{args.source_domain}.pt")))
    classifier_teacher = feat_classifier(args.class_num).cuda()
    classifier_teacher.load_state_dict(torch.load(os.path.join(args.weight_basepath,f"classifier_{args.source_domain}.pt")))
    resnet_teacher.eval()
    classifier_teacher.eval()

    # init test
    acc_best = 0

    # other
    PG = Logic_Save(args,resnet_teacher,classifier_teacher,target_dataloader)

    mixup = v2.MixUp(num_classes=args.class_num)

    # train
    Project.log(f"======> Transfer {args.source_domain}->{args.target_domain} <=======")
    for now_epoch in range(args.epoch):
        for data in target_dataloader['train']:
            model.train()

            index = data['index'].cuda()
            img = data['img'].cuda()
            train_cls_transformcon = data['train_cls_transformcon'].cuda()
            train_transforms_1 = data['train_transforms_1'].cuda()
            train_transforms_2 = data['train_transforms_2'].cuda()
            groundTruth = data['label'].cuda()



            # no grad update storage & selection
            with torch.no_grad():
                clear_feature = resnet_teacher(img)
                clear_logic = torch.softmax(classifier_teacher(clear_feature), dim=1)  
            neibor_logic = PG.get_logic(index,"neighbor_entropy_K")     # 邻居K标签融合
            neibor_logic_label = torch.argmax(neibor_logic, dim=1)      # 邻居K标签

            ### 基于置信度变化的样本筛选
            prior_logic = PG.get_logic(index, "raw")
            confidence_change = get_confidence_change(index, clear_logic, prior_logic, args.confidence_change_tau)      # 计算置信度变化, 注意: epoch=0时所有都是confidence_fluctuate
            confidence_stable = confidence_change['confidence_stable_inter']                # 稳定样本
            confidence_fluctuate = confidence_change['confidence_fluctuate_inter']          # 波动样本

            ################
            ###   Loss   ###
            ################

            # optim
            predict_1, projection_2, output_logic = model(train_transforms_1, train_transforms_2, train_cls_transformcon)   # 对比学习
            loss_ctr = CTRR_Loss(predict_1, projection_2, output_logic, args.CTRR_tau)

            ### MIXUP 保证分类器平滑性——对波动样本处理；本版本是8976
            mix_img, mix_softlabel = mixup(img, torch.argmax(neibor_logic, dim=1))
            mix_logic = torch.softmax(model.forward_test(mix_img),dim=1)
            mixup_ce_loss = MIXUP_CE(mix_logic,mix_softlabel)


            ### hard
            if confidence_fluctuate.shape[0] > 1:
                hard_train_img = train_cls_transformcon[confidence_fluctuate]
                hard_train_logic = neibor_logic[confidence_fluctuate]
                hard_train_pesudo = neibor_logic_label[confidence_fluctuate]
                hard_train_pesudo_onehot = nn.functional.one_hot(hard_train_pesudo, args.class_num).float()
                hard_logic = torch.softmax(model.probability(hard_train_img),dim=1)
                hard_logic_2 = torch.softmax(model.probability(train_transforms_1[confidence_fluctuate]),dim=1)
                # fluctuate_loss = MSE(hard_logic, hard_train_pesudo_onehot)                # k 聚类硬标签-实验记录序号11【8955】
                # fluctuate_loss = MSE(hard_logic, hard_train_logic)                        # k 聚类软标签-实验记录序号12【8955】
                fluctuate_loss = MSE(hard_logic, hard_logic_2)                              # 两次预测取一致性【8955】

                
            else:
                fluctuate_loss = torch.tensor([0.0]).cuda()    

            ### CE 保证学习到语义信息——对稳定样本处理
            if confidence_stable.shape[0] > 1:
                stable_logic = torch.softmax(model.probability(img[confidence_stable]),dim=1)
                stable_pseudo = torch.argmax(neibor_logic, dim=1)
                CE_loss = CE(stable_logic, stable_pseudo[confidence_stable])
            else:
                CE_loss = torch.tensor([0.0]).cuda()

            loss = args.CTRRloss_weight * loss_ctr +  mixup_ce_loss + CE_loss + fluctuate_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # other
            PG.update(clear_feature, clear_logic, index, now_epoch)

        # update
        resnet_teacher,classifier_teacher = ema_update_CTRR(model,resnet_teacher,classifier_teacher,args.EMA)
        # test
        teacher_Acc = test_domain(resnet_teacher,classifier_teacher,target_dataloader['test'])


        if acc_best < teacher_Acc:
            acc_best = teacher_Acc
        log_str = f"[Train {args.dataset} {args.source_domain}->{args.target_domain}] Epoch {now_epoch:>3}/{args.epoch:>3} teacher resnet: {teacher_Acc:.4f} Acc Best:{acc_best:.4f}"
        Project.log(log_str)


def SFDA_tar(args, dataset_dirt): 
    args.epoch = 80
    args.batch_size = 16
    args.lr = 1e-3
    args.confidence_change_tau =  0.025
    args.CTRR_tau = 0.9
    args.CTRRloss_weight = 100
    args.topk = 10
    args.EMA = 0.9
    args.fluc_weight = 1
    target_transfer(args, dataset_dirt)






