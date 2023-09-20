from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np


def valid(valid_loader, epoch, cam):
    cam.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    total_loss, total_correct, total_samples = 0, 0, 0

    # Each batch
    for X, labels in valid_loader:
        labels = labels.cuda()

        with torch.no_grad():
            lista_rep = np.array([np.array(X[:, i, :, :].transpose(1,2)) for i in range(X.shape[1])])
            
            lista_rep = torch.tensor(lista_rep, dtype=torch.float32).cuda()
            final_outs = cam(lista_rep).squeeze()
            if len(final_outs.shape) == 1:
                final_outs = final_outs.view(1, -1)
            loss = criterion(final_outs, labels).cuda()

        total_loss += loss.item()
        _, predicted = torch.max(final_outs, 1)
        _, gt = torch.max(labels, 1)
        total_correct += (predicted == gt).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(valid_loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy