import sys, os
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.model_selection import train_test_split
import numpy as np, glob, random, pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from cam import CAM
from training import train
from testing import test

pca_size = 150
nro_rep = 10
epochs = 50
SEED = 0
dataset_target = 'CK+' #'JAFFE' 


if (SEED == 0):
	torch.backends.cudnn.benchmark = True
else:
	print("Using SEED")
	random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(SEED)

class CustomDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data = data_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        return image, label


def fit_evaluate_model_CV(X_train, y_train, X_test, y_test):

    X_train = torch.tensor(np.array(X_train)[:,:nro_rep,:,:], dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test)[:,:nro_rep,:,:], dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    print("Number of Train samples:" + str(len(train_dataset)))
    print("Number of Val samples:" + str(len(test_dataset)))

    cam = CAM(nro_rep).cuda()
    optimizer = torch.optim.Adam(cam.parameters(), 0.001)

    
    for epoch in range(epochs):
        Training_loss, Training_acc = train(train_loader, optimizer, epoch, cam)
        Test_loss, Test_acc = test(test_loader, epoch, cam)
        torch.save(cam.state_dict(), "model_cam.pth")
    return Test_loss, Test_acc

if dataset_target == 'CK+':
    lista_repr_paths = glob.glob('../temp2/CK/10 REP/L/*')
    y = np.load('../y_ck.npy')
    y = y-1

if dataset_target == 'JAFFE':
    lista_repr_paths = glob.glob('../temp2/JAFFE/10 REP/L/*')
    y = np.load('../y.npy')

# Load representations
lista_rep = []
for rep_path in lista_repr_paths:
    lista_rep.append(np.load(rep_path))

# Reduce dimensionality to 150
LX = [] 
for i in range(0, len(lista_rep)):
    pca = PCA(n_components=pca_size)
    X = pca.fit(lista_rep[i]).transform(lista_rep[i])
    LX.append(X)


final_rep = LX[0]
for i in range(len(lista_rep)-1):
    final_rep = np.hstack((final_rep, LX[i+1]))

X = np.resize(final_rep, (final_rep.shape[0],final_rep.shape[1]//pca_size,pca_size, 1))
y = [int(a) for a in y.squeeze()]
y = torch.nn.functional.one_hot(torch.tensor(y))


acc_per_fold, loss_per_fold, acc_per_fold_no_attention, loss_per_fold_no_attention = [], [], [], []
acc_per_fold_rf_loaded, acc_per_fold_rf_base = [], []

subject_index = 0
subjects = []

if dataset_target == 'CK+':
    labeled_path_2 = "/dataset/CKold/cohn-kanade-images/"
if dataset_target == 'JAFFE':
    labeled_path_2 = "/dataset/jaffe/"

for participant in os.listdir(os.path.join(labeled_path_2)):
    subjects.append(subject_index)
    for sequence in os.listdir(os.path.join(labeled_path_2, participant)):
        if sequence != ".DS_Store":
            subject_index += 1

for i, subject in enumerate(subjects):
    print(f"\nSubject:{i}, Offset: {subject}")
    # Define models
    X_train, y_train, X_test, y_test = None, None, None, None 
    if i == len(subjects) - 1:
        X_train = X[0:subject]
        y_train = y[0:subject]
        X_test = X[subject:]
        y_test = y[subject:]
    else:
        length = subjects[i + 1] - subjects[i]
        X_train = np.vstack((X[0:subject], X[subject + length:]))
        y_train = np.vstack((y[0:subject], y[subject + length:]))
        X_test = X[subject:subject + length]
        y_test = y[subject:subject + length]

    
    # NN models
    test_loss, test_acc = fit_evaluate_model_CV(X_train, y_train, X_test, y_test)
    
    print(f"Attention acc: {test_acc}")
    acc_per_fold.append(test_acc * 100)
    loss_per_fold.append(test_loss)
    

print('------------------------------------------------------------------------')
print('Average scores for all folds with CNN Attention:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')