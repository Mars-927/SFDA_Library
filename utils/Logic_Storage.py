import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.Project_Record import Project


def entropy(predictions,dim=1):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=dim)
    return H


class Logic_Save():
    def __init__(self,args,resnet,classifier,dataloader):
        resnet.eval()
        classifier.eval()
        self.predict_time_storage = torch.zeros(args.train_dataset_size, args.epoch+1, args.class_num).cuda()         # predict存储, 尺寸为-[epoch, dataset_size, class_num]
        self.predict_line_storage = torch.zeros(args.train_dataset_size, args.class_num).cuda()                      # predict存储, 尺寸为-[dataset_size, class_num]
        self.dataset_feature = torch.zeros(args.train_dataset_size, 256).cuda()             # 特征存储  
        self.dataset_groundtruth = torch.zeros(args.train_dataset_size).long().cuda()       # GroundTruth存储
        self.dataset_path = [ 0 for _ in range(args.train_dataset_size) ]
        self.class_num = args.class_num
        self.neighbor_topk = 10
        self.eval_dict = dict()

        with torch.no_grad():
            for item in dataloader['all']:  # !!!only train == all use
                img = item['img'].cuda()
                index = item['index'].cuda()
                groundtruth = item['label'].cuda()
                path = item['path']
                output_feature = resnet(img)
                output_logic = torch.softmax(classifier(output_feature),dim=1)
                output_feature_norm = F.normalize(output_feature,dim=1)
                self.predict_time_storage[index, 0, :] = output_logic.clone()            # 注意,使用predict_time_storage时指针为epoch + 1
                # self.predict_line_storage[index] = output_logic.clone()
                self.dataset_feature[index] = output_feature_norm.clone()
                self.dataset_groundtruth[index] = groundtruth.clone()
                for sample,value in enumerate(index):
                    self.dataset_path[value] = path[sample]

        resnet.train()
        classifier.train()

    def update(self,feature, logic, index, epoch):
        # 输入的内容务必为softmax后的结果
        feature = F.normalize(feature,dim=1)
        self.dataset_feature[index] = feature.clone()
        self.predict_time_storage[index, epoch+1, :] = logic.clone()
        self.predict_line_storage[index, :] = logic.clone()

    def get_logic(self,img_index,mode):
        if mode == "raw":
            return self.predict_line_storage[img_index]
        
        elif mode == "neighbor_K":
            query_feature = self.dataset_feature[img_index]
            dataset_feature = self.dataset_feature
            cosin_distance = F.cosine_similarity(query_feature.unsqueeze(1), dataset_feature.unsqueeze(0), dim=2)       # 计算余弦相似度,得到[query_size, dataset_size]
            neighbor_indicate = torch.topk(cosin_distance,k = self.neighbor_topk, dim=1 , largest=True)[1]              # 计算topk的样本下标,得到[query, topk]
            neighbor_indicate = torch.flatten(neighbor_indicate)                                                        # 展平便于取样本,得到[query * topk]
            tmp = self.predict_line_storage[neighbor_indicate,:]                                                        # 选取样本logic,得到[query * topk, class_num]
            neighbor_result = tmp.reshape( img_index.shape[0], self.neighbor_topk, self.class_num )                     # 整形,得到[query, topk, class_num]
            query_logic = torch.squeeze(torch.sum(neighbor_result,dim=1))                                               # 邻居维度求和,得到[query,class_num]
            query_logic = torch.softmax(query_logic, dim=1)                                                             # 概率化处理
            return query_logic
        
        elif mode == "neighbor_entropy_K":
            query_feature = self.dataset_feature[img_index]
            dataset_feature = self.dataset_feature
            cosin_distance = F.cosine_similarity(query_feature.unsqueeze(1), dataset_feature.unsqueeze(0), dim=2)       # 计算余弦相似度,得到[query_size, dataset_size]
            neighbor_indicate = torch.topk(cosin_distance,k = self.neighbor_topk, dim=1 , largest=True)[1]              # 计算topk的样本下标,得到[query, topk]
            neighbor_indicate = torch.flatten(neighbor_indicate)                                                        # 展平便于取样本,得到[query * topk]
            tmp = self.predict_line_storage[neighbor_indicate,:]                                                             # 选取样本logic,得到[query * topk, class_num]
            neighbor_result = tmp.reshape( img_index.shape[0], self.neighbor_topk, self.class_num )                     # 整形,得到[query, topk, class_num]

            # caculate weight
            w = entropy(neighbor_result,dim=2)                                                                          # 每个预测的熵,得到[query, topk]
            w = w / torch.log2(torch.tensor(self.class_num))
            neighbor_weight = torch.exp(-w)
            weighted_result = neighbor_weight.unsqueeze(dim=2) * neighbor_result                                        # 有熵的预测,得到[query, topk, class_num]
            query_logic = torch.squeeze(torch.sum(weighted_result,dim=1))                                               # 邻居维度求和,得到[query,class_num]
            query_logic = torch.softmax(query_logic, dim=1)                                                             # 概率化处理
            return query_logic

    def storage_eval(self, mode):
        if mode == "raw":
            storage_pseudo = torch.argmax(self.predict_line_storage, dim = 1)
            tmp = (self.dataset_groundtruth == storage_pseudo).float().cpu().tolist()
            raw_acc = np.mean(tmp)
            return raw_acc

        elif mode == "neighbor_K":
            num_range = torch.arange(start = 0,end = self.dataset_feature.shape[0])
            all_mix_logic = self.get_logic(num_range,"neighbor_K")
            all_mix_pseudo = torch.argmax(all_mix_logic, dim=1)
            tmp = (self.dataset_groundtruth == all_mix_pseudo).float().cpu().tolist()
            mix_acc = np.mean(tmp)
            return mix_acc
        
        elif mode == "neighbor_entropy_K":
            num_range = torch.arange(start = 0,end = self.dataset_feature.shape[0])
            all_mix_logic = self.get_logic(num_range,"neighbor_entropy_K")
            all_mix_pseudo = torch.argmax(all_mix_logic, dim=1)
            tmp = (self.dataset_groundtruth == all_mix_pseudo).float().cpu().tolist()
            mix_acc = np.mean(tmp)
            return mix_acc
        
    def storage_eval_draw(self, writer, now_epoch):
        # 绘制本地图像, 三个一折线图
        plt.clf()
        modes = ["raw", "neighbor_K", "neighbor_entropy_K"]
        for item in modes:
            if item not in self.eval_dict.keys():
                self.eval_dict[item] = []
            now_acc = self.storage_eval(item)
            self.eval_dict[item].append(now_acc)
            writer.add_scalar(f'Pseudo Acc\{item}', now_acc, now_epoch)

        img_path = os.path.join(Project.root_path,"storage_acc.png")
        _, ax = plt.subplots(figsize=(8, 6))
        for key,value in self.eval_dict.items():
            x = list(range(1,len(value)+1))
            y = self.eval_dict[key]
            ax.plot(x, y, label=key)
            ax.legend()
            ax.set_title('Storage Acc')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('ACC')
        plt.savefig(img_path, dpi=300)
        plt.close()

        # 绘制tensorborad图像, 三个一折线图



