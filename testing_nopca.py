from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from itertools import accumulate

def test(test_loader, cam, lista_shapes):
    cam.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    total_loss, total_correct, total_samples = 0, 0, 0
    array_acumulado = list(accumulate(lista_shapes))

    # Each batch
    for X, labels in test_loader:
        labels = labels.cuda()

        with torch.no_grad():
            lista_rep = [torch.tensor(np.expand_dims(X[:,:array_acumulado[0]],axis=1), dtype=torch.float32).cuda()]
            for i in range(len(lista_shapes)-1):
                lista_rep.append(torch.tensor(np.expand_dims(X[:,array_acumulado[i]:array_acumulado[i+1]],axis=1), dtype=torch.float32).cuda())

            final_outs = cam(lista_rep).squeeze()
            if len(final_outs.shape) == 1:
                final_outs = final_outs.view(1, -1)
            loss = criterion(final_outs, labels).cuda()

        total_loss += loss.item()
        _, predicted = torch.max(final_outs, 1)
        _, gt = torch.max(labels, 1)
        total_correct += (predicted == gt).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy