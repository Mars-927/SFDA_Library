import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.loss import SoftTargetCrossEntropy

def predict_discrepancy(out1,out2):
    return torch.mean(torch.abs(torch.softmax(out1,dim=1) - torch.softmax(out2,dim=1)))



class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
def Entropy(predictions):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    return H.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size                                            
        self.register_buffer("temperature", torch.tensor(temperature).cuda())			                                    # 超参数-温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float())		# 主对角线为0，其余位置全为1的mask矩阵,2*batch x 2*batch
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到[输入图像应该是两个特征,即[batch,dim]]
        z_i = nn.functional.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)【归一化】
        z_j = nn.functional.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)【z_i,z_j】

        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    
def contrast_loss(feature_A,feature_B):
    CT_norm = ContrastiveLoss(batch_size=feature_A.shape[0])
    CT_loss = CT_norm(feature_A,feature_B)
    return CT_loss


def entropy_loss(logic):
    # 个体熵损失——降低个体熵
    epsilon = 1e-8
    H = -logic * torch.log(logic + epsilon)
    H = H.sum(dim=1)
    return H.mean()

def diversity_loss(logic):
    # batch熵损失——最大化群体熵
    epsilon=1e-8
    probs_mean = logic.mean(dim=0)
    loss_div = - torch.sum(- probs_mean * torch.log(probs_mean + epsilon))
    return loss_div

def classifier_avg_loss(classifier_head,class_num):
    K = class_num
    norm_classifier_head = F.normalize(classifier_head,dim=1)
    similarity_matrix = 1 + F.cosine_similarity(norm_classifier_head.unsqueeze(1), norm_classifier_head.unsqueeze(0), dim=2)
    similarity_matrix = similarity_matrix / 2
    loss = torch.sum(similarity_matrix**2) / K / (K-1)
    loss.backward()
    return loss

def CTRR_Loss(predict_1, projection_2, output_logic, tau):
    batch_size = predict_1.shape[0]
    predict_1 = torch.clamp(predict_1, 1e-4, 1.0 - 1e-4)          # avoid collapsing and gradient explosion
    projection_2 = torch.clamp(projection_2, 1e-4, 1.0 - 1e-4)
    contrast_1 = torch.matmul(predict_1, projection_2.t())           # B X B
    contrast_1 = -contrast_1*torch.zeros(batch_size, batch_size).fill_diagonal_(1).cuda() + ((1-contrast_1).log()) * torch.ones(batch_size, batch_size).fill_diagonal_(0).cuda()
    contrast_logits = 2 + contrast_1
    soft_targets = torch.softmax(output_logic, dim=1)
    contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()
    contrast_mask.fill_diagonal_(1)
    pos_mask = (contrast_mask >= tau).float()
    contrast_mask = contrast_mask * pos_mask
    contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True)
    loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)
    return loss_ctr

def nl_criterion(output, y, num_class):
    output = torch.log( torch.clamp(1.-F.softmax(output, dim=1), min=1e-5, max=1.) )
    labels_neg = ( (y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, num_class).cuda()) % num_class ).view(-1)
    l = F.nll_loss(output, labels_neg, reduction='none')
    return l

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, probabilities):
        # 确保输入是一个概率分布
        if not torch.all(probabilities >= 0) or not torch.all(probabilities <= 1):
            raise ValueError("Input probabilities must be in the range [0, 1].")
        
        # 计算熵
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # 加上小常数以避免log(0)
        return torch.mean(entropy)



CE = nn.CrossEntropyLoss()
MMD = MMDLoss()
MSE = nn.MSELoss()
CONS = contrast_loss
MIXUP_CE = SoftTargetCrossEntropy()
NLCE = nl_criterion
ENloss = EntropyLoss()