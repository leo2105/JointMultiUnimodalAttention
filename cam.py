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
        # self.affine_1 = nn.Linear(1, 1, bias=False)
        # self.affine_2 = nn.Linear(1, 1, bias=False)
            self.arr_W.append(nn.Linear(1, 32, bias=False).cuda())
        # self.W_1 = nn.Linear(1, 32, bias=False)
        # self.W_2 = nn.Linear(1, 32, bias=False)
            self.arr_W_c.append(nn.Linear(150*self.nro_rep, 32, bias=False).cuda())
        # self.W_c1 = nn.Linear(300, 32, bias=False)
        # self.W_c2 = nn.Linear(300, 32, bias=False)
            self.arr_W_h.append(nn.Linear(32, 1, bias=False).cuda())
        # self.W_h1 = nn.Linear(32, 1, bias=False)
        # self.W_h2 = nn.Linear(32, 1, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.classificator = nn.Sequential(nn.Linear(150*nro_rep, 128),
                                           nn.Dropout(0.6),
                                           nn.Linear(128,7))

    #def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    #def forward(self, feat_1, feat_2):
    def forward(self, feats_list):
        arr_afines = []
        arr_t = []
        arr_att = []
        arr_H = []
        arr_att_features = []

        #feat_1_2 = torch.cat((feat_1, feat_2),2)
        #feat_n = torch.cat(feats_list, 3)
        feat_n = feats_list.view(feats_list.size()[1], feats_list.size()[2], -1)
        for i in range(self.nro_rep):
            arr_t.append(self.arr_affines[i](feat_n.transpose(1,2)))
        # t_1 = self.affine_1(feat_1_2.transpose(1,2))
        # t_2 = self.affine_2(feat_1_2.transpose(1,2))
            att_aux = torch.matmul(feats_list[i].transpose(1,2), arr_t[i].transpose(1,2))
            arr_att.append(self.tanh(torch.div(att_aux,  math.sqrt(feat_n.shape[1]))))
        #att_1 = torch.matmul(feat_1.transpose(1,2), t_1.transpose(1,2))
        #att_1 = self.tanh(torch.div(att_1,  math.sqrt(feat_1_2.shape[1])))
        
        # att_2 = torch.matmul(feat_2.transpose(1,2), t_2.transpose(1,2))
        # att_2 = self.tanh(torch.div(att_2, math.sqrt(feat_1_2.shape[1])))

            a = self.arr_W_c[i].cuda()(arr_att[i])
            b = self.arr_W[i](feats_list[i].transpose(1,2))
            arr_H.append(a + b)
            arr_H[i] = self.relu(arr_H[i])
        # H_1 = self.relu(self.W_c1(att_1) + self.W_1(feat_1.transpose(1,2)))
        # H_2 = self.relu(self.W_c2(att_2) + self.W_2(feat_2.transpose(1,2)))
            arr_att_features.append(self.arr_W_h[i](arr_H[i]).transpose(1,2) + feats_list[i])
        # att_1_features = self.W_h1(H_1).transpose(1,2) + feat_1
        # att_2_features = self.W_h2(H_2).transpose(1,2) + feat_2

        features_final = torch.cat(arr_att_features, 2)
        # features_1_2 = torch.cat((att_1_features, att_2_features), 2)
        outs = self.classificator(features_final) #.transpose(0,1))
        # outs = self.classificator(features_1_2) #.transpose(0,1))

        return outs 
