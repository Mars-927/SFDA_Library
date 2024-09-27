import torch
from torch import nn
import torch.nn.functional as F

class AdaMoCo(nn.Module):
    def __init__(self, src_model, momentum_model, features_length, num_classes, dataset_length, temporal_length):
        super(AdaMoCo, self).__init__()
        self.m = 0.999                                  # 动量更新
        self.first_update = True                        # 首次更新
        self.src_model = src_model                      # 初始模型
        self.momentum_model = momentum_model            # moco用额外模型
        self.momentum_model.requires_grad_(False)       # moco额外模型停止梯度

        self.queue_ptr = 0                  # 队列指针
        self.mem_ptr = 0                    # 队列指针
        self.T_moco = 0.07

        #queue length
        self.K = min(16384, dataset_length)         # 队列大小
        self.memory_length = temporal_length        # 存储大小

        # 分配固定空间用于存储队列
        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer("labels", torch.randint(0, num_classes, (self.K,)))
        self.register_buffer("idxs", torch.randint(0, dataset_length, (self.K,)))
        self.register_buffer("mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length)))
        self.register_buffer("real_labels", torch.randint(0, num_classes, (dataset_length,)))
        self.features = F.normalize(self.features, dim=0)
        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder_q -> encoder_k 动量更新模型
        for param_q, param_k in zip(self.src_model.parameters(), self.momentum_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)


    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        # 更新moco使用
        start = self.queue_ptr                                      # moco 队列开始指针            
        end = start + len(keys)                                     # moco 队列结束指针
        idxs_replace = torch.arange(start, end).cuda() % self.K     # idx 转换符合队列长度
        self.features[:, idxs_replace] = keys.T                     # 特征
        self.labels[idxs_replace] = pseudo_labels                   # 标签
        self.idxs[idxs_replace] = idxs                              # idx
        self.real_labels[idxs_replace] = real_label                 # 真实标签
        self.queue_ptr = end % self.K                               # 队列 ptr
        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels         # 队列伪标签
        self.mem_ptr = epoch % self.memory_length                   # memory 指针


    @torch.no_grad()
    def get_memory(self):
        # 获取memory存储
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False):
        # moco得到正例和负例1
        # compute query features
        feats_q, logits_q = self.src_model(im_q)        # 计算特征

        if cls_only:                                    # 只用来分类
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)                 # 对于query 特征进行归一化

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()         # 更新key encoder
            k, _ = self.momentum_model(im_k)        
            k = F.normalize(k, dim=1)

        # compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)                  # 正例的相似度 positive logits: Nx1
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])  # 负例的相似度 negative logits: NxK
        logits_ins = torch.cat([l_pos, l_neg], dim=1)                           # 拼接逻辑值 logits: Nx(1+K)
        logits_ins /= self.T_moco                                               # 应用温度参数

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k
