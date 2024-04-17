'''
File Name: models
Create File Time: 2024/4/18 00:04
File Create By Author: Yang Guanqun
Email: yangguanqun01@corp.netease.com
Corp: Fuxi Tech, Netease
'''
import torch.nn as nn

class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)

    def forward(self, input):
        x = F.normalize(input, dim=-1) * input.shape[-1] ** 0.5
        return self.linear(x)

