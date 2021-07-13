import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


class DistillationLoss(torch.nn.Module):
    def __init__(self, temp: float):
        super(DistillationLoss, self).__init__()
        self.T = temp

    def forward(self, out1, out2):
        loss = F.kl_div(
            F.log_softmax(out1 / self.T, dim=0),
            F.softmax(out2 / self.T, dim=0),
            reduction="none",
        )

        return loss


class Pair_loss(torch.nn.Module):
    def __init__(self):
        super(Pair_loss, self).__init__()
        self.kd_criterion = DistillationLoss(temp=1)
        self.mmd_criterion = torch.nn.MSELoss(reduction="none")

    def forward(self, scores, scores2):
        kd_loss = self.kd_criterion(scores,scores2) + self.kd_criterion(scores2,scores)
        mmd_loss = self.mmd_criterion(scores,scores2)

        return kd_loss + mmd_loss

class SDAD_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(SDAD_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        #print(f'score:{score.shape}')
        loss_cls = self.criterion(score, label)

        score_normal_max=score_normal.max(dim=-1)[0]
        score_abnormal_max=score_abnormal.max(dim=-1)[0]
        hinge_loss=1-score_abnormal_max+score_normal_max
        hinge_loss=torch.max(hinge_loss,torch.zeros_like(hinge_loss)).mean()

        return loss_cls+hinge_loss


def train(nloader, aloader, model, batch_size, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
        #features of original and its mirror transformation
        ninput,ninput2 = torch.split(ninput,32,dim=2)
        ainput,ainput2 = torch.split(ainput,32,dim=2)

        input = torch.cat((ninput, ainput), 0).to(device)
        input2 = torch.cat((ninput2,ainput2),0).to(device)

        #some returns aren't used
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores = model(input)  # b*32  x 2048
        score_abnormal2, score_normal2, feat_select_abn2, feat_select_normal2, scores2 = model(input2)

        scores = scores.view(batch_size * 32 * 2, -1)
        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]
        scores2 = scores2.view(batch_size * 32 * 2, -1)
        scores2 = scores2.squeeze()
        abn_scores2 = scores2[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        #SDAD_loss consists of MIL ranking loss and BCE loss
        loss_criterion = SDAD_loss(0.0001, 100)
        #Pair_loss consists of MSE loss and KL loss (= kd-> knowledge-distillation)
        pair_criterion = Pair_loss()

        #MIL sparsity loss
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)+sparsity(abn_scores2,batch_size,8e-3)#defatul:8e-3
        loss_smooth = smooth(abn_scores, 8e-4)+smooth(abn_scores2,8e-4)#default:8e-4
        bce_loss=loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_criterion(score_normal2, score_abnormal2, nlabel, alabel, feat_select_normal2, feat_select_abn2)
        pair_loss=torch.mean(pair_criterion(scores,scores2))
        
        #final loss
        cost = bce_loss +pair_loss + loss_smooth + loss_sparse 

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


