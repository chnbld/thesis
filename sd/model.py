import torch
import torch.nn as nn
import torch.nn.init as torch_init
import numpy as np
from kmeans_pytorch import kmeans
from scipy.spatial import distance
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Model(nn.Module):
    def __init__(self, n_features=32, batch_size=16):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = 32
        self.k_nor = 32#self.num_segments // 10

        #self.Aggregate = Aggregate(len_feature=5120)
        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    
    def forward(self, inputs):
        #'inputs:{inputs.shape}')
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()
        
        out = out.view(-1, t, f)

        out = self.drop_out(out)

        features = out
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)
        #print(f'inputs.shape:{inputs.shape[0]}')
        if inputs.shape[0]!=32:
            return scores

        normal_features = features[0:self.batch_size]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size:]
        abnormal_scores = scores[self.batch_size:]
        
        n_size = normal_features.shape[0]

        #######  process abnormal videos

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        score_abnormal = torch.mean(abnormal_scores, dim=1)  

        ####### process normal videos

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        score_normal = torch.mean(normal_scores, dim=1)

        feat_select_abn = abnormal_features.squeeze()
        feat_select_normal = normal_features.squeeze()
        
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores
