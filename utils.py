import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import normalize

class TripletLoss(nn.Module):
    def __init__(self,margin=0.8):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative,alpha,beta):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(self.margin + alpha*distance_positive - beta*distance_negative, min=0.0)
        return torch.mean(loss)


class PatchCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PatchCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        inputs: B*196
        targets: B*196
        """
        batch_size, num_patches = inputs.size()

        loss_matrix = torch.zeros(batch_size, num_patches, device=inputs.device)

        for i in range(num_patches):
            loss_fn = nn.CrossEntropyLoss()
            loss_matrix[:, i] = loss_fn(inputs[:, i].unsqueeze(1), targets[:, i].long())

        return loss_matrix

class Arc(nn.Module):
    def __init__(self, feat_num, cls_num) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn((feat_num, cls_num))) 

    def forward(self, x, m=1, s=10):
        x_norm = normalize(x, p=2, dim=1) # [N, 2]
        w_norm = normalize(self.w, p=2, dim=0) # [2, 10]
        cosa = torch.matmul(x_norm, w_norm) / s 
        a = torch.arccos(cosa) 
        top = torch.exp(s*torch.cos(a+m)) # [N, 10]
        down = top + torch.sum(torch.exp(s*cosa), dim=1, keepdim=True) - torch.exp(s*cosa)
        arc_softmax = top/(down+1e-10)
        return arc_softmax

class Instance_CE(nn.Module):
    def __init__(self):
        super(Instance_CE, self).__init__()

    def forward(self,patch_prediction , pseudo_instance_label):#pred:B*196*2,target B*196
        patch_prediction = patch_prediction.float()
        pseudo_instance_label = pseudo_instance_label.float()

        patch_prediction = torch.softmax(patch_prediction, dim=-1)

        loss_student = -1. * torch.mean(
            (1 - pseudo_instance_label) * torch.log(patch_prediction[:, :, 0] + 1e-8) +
            pseudo_instance_label * torch.log(patch_prediction[:, :, 1] + 1e-8)
        )

        return loss_student

def instance_CE(patch_prediction,pseudo_instance_label):#pred:B*196*2,target B*196
    patch_prediction = patch_prediction.float()
    pseudo_instance_label = pseudo_instance_label.float()
    patch_prediction = torch.sigmoid(patch_prediction)
    loss_student = -1. * torch.mean((1 - pseudo_instance_label.squeeze()) * torch.log(patch_prediction[:,:, 0] + 1e-8) +
                              pseudo_instance_label.squeeze()* torch.log(patch_prediction[:,:, 1] + 1e-8))
    return loss_student