from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

class CAM(nn.Module):
    def __init__(self, nro_rep):
        super(CAM, self).__init__()

        self.nro_rep = nro_rep
        self.arr_affines = []
        self.arr_W = []
        self.arr_W_c = []
        self.arr_W_h = []

        for i in range(self.nro_rep):
            self.arr_affines.append(nn.Linear(1, 1, bias=False).cuda())
            self.arr_W.append(nn.Linear(1, 32, bias=False).cuda())
            self.arr_W_c.append(nn.Linear(150*self.nro_rep, 32, bias=False).cuda())
            self.arr_W_h.append(nn.Linear(32, 1, bias=False).cuda())

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.classificator = nn.Sequential(nn.Linear(150*nro_rep, 128),
                                           nn.Dropout(0.6),
                                           nn.Linear(128,7))

    def forward(self, feats_list):
        arr_afines = []
        arr_t = []
        arr_att = []
        arr_H = []
        arr_att_features = []

        feat_n = feats_list.view(feats_list.size()[1], feats_list.size()[2], -1)
        for i in range(self.nro_rep):
            arr_t.append(self.arr_affines[i](feat_n.transpose(1,2)))
            att_aux = torch.matmul(feats_list[i].transpose(1,2), arr_t[i].transpose(1,2))
            arr_att.append(self.tanh(torch.div(att_aux,  math.sqrt(feat_n.shape[1]))))

            a = self.arr_W_c[i].cuda()(arr_att[i])
            b = self.arr_W[i](feats_list[i].transpose(1,2))
            arr_H.append(a + b)
            arr_H[i] = self.relu(arr_H[i])
            arr_att_features.append(self.arr_W_h[i](arr_H[i]).transpose(1,2) + feats_list[i])

        features_final = torch.cat(arr_att_features, 2)
        outs = self.classificator(features_final)

        return outs 
