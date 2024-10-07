from torch import nn
from model.res_network import resnet, feat_classifier
import torch,os

class BatchNorm1d(nn.Module):
    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)
    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class Linear(nn.Module):
    def __init__(self, nb_classes=10, feat=512):
        super(Linear, self).__init__()
        self.linear = nn.Linear(feat, nb_classes)
    def forward(self, x):
        return self.linear(x)

class SimSiam(nn.Module):
    def __init__(self, args):
        super(SimSiam, self).__init__()
        self.emam = 0.99
        self.class_num = args.class_num
        dim_in = 256
        dim_out = args.class_num
        dim_mlp = 2048

        self.backbone = resnet().cuda()
        self.backbone_k = resnet().cuda()
        self.classifier = feat_classifier(args.class_num).cuda()
        self.backbone.load_state_dict(torch.load(os.path.join(args.weight_basepath,f"resnet_{args.source_domain}.pt"))) 
        self.backbone_k.load_state_dict(torch.load(os.path.join(args.weight_basepath,f"resnet_{args.source_domain}.pt")))
        self.classifier.load_state_dict(torch.load(os.path.join(args.weight_basepath,f"classifier_{args.source_domain}.pt"))) 
    
        # 映射头由两个线性层组成
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_mlp), 
            BatchNorm1d(dim_mlp), 
            nn.ReLU(), 
            nn.Linear(dim_mlp, dim_out)
            )
        self.projector_k = nn.Sequential(
            nn.Linear(dim_in, dim_mlp), 
            BatchNorm1d(dim_mlp), 
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
            )
        self.encoder = nn.Sequential(
            self.backbone,                  # 编码器
            self.projector                  # 投影层
        )
        self.encoder_k = nn.Sequential(
            self.backbone_k,
            self.projector_k
        )

        # 动量更新
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False       # 【下分支没有梯度】

        # 预测头——概率预测的两个分支之
        self.predictor = nn.Sequential(     # 预测层
            nn.Linear(dim_out, 512), 
            BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Linear(512, dim_out)
            )
        
        # 来源于图5——用来对图像进行预测计算CE损失
        self.linear = Linear(nb_classes=self.class_num, feat=dim_in)
        self.probability = nn.Sequential(
            self.backbone,
            self.classifier,
        )

    def forward(self, image_aug1, image_aug2, image_weak):
        logic = self.probability(image_weak)

        p1 = nn.functional.normalize(self.predictor(self.encoder(image_aug1)), dim=1)               # 编码层 + 映射层 得到z 【上分支】，使用预测层得到 p
        z2 = nn.functional.normalize(self.encoder_k(image_aug2), dim=1)                             # 编码层 + 映射层 得到z 【下分支】

        # ema更新到 编码层 + 映射层, 由于K分支没有梯度, 因此使用q分支更新
        with torch.no_grad():
            m = self.emam
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

        return p1,z2,logic

    def forward_test(self, x):
        # 用于预测测试的头,这里优化的是特征提取器
        return self.probability(x)