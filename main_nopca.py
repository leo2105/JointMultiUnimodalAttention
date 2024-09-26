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
from cam_nopca import CAM
from training_nopca import train
from testing_nopca import test
from validation_nopca import valid


pca_size = int(os.getenv('PCA_SIZE', 150))
epochs = int(os.getenv('NRO_REP', 10))
nro_rep = int(os.getenv('EPOCHS', 50))
kind_rep = os.getenv('KIND_REP', 'L')
dataset_target = os.getenv('DATASET_TARGET', 'JAFFE')
folder_path_rep = os.getenv('FOLDER_PATH_REP', 'temp2')

SEED = 0

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


def fit_evaluate_model_CV(X_train, y_train, X_test, y_test, lista_shapes):

    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    X_train_ = torch.tensor(np.array(X_train_), dtype=torch.float32)
    y_train_ = torch.tensor(np.array(y_train_), dtype=torch.float32)
    X_valid_ = torch.tensor(np.array(X_valid_), dtype=torch.float32)
    y_valid_ = torch.tensor(np.array(y_valid_), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    train_dataset = CustomDataset(X_train_, y_train_)
    valid_dataset = CustomDataset(X_valid_, y_valid_)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    print("Number of Train samples:" + str(len(train_dataset)))
    print("Number of Valid samples:" + str(len(valid_dataset)))
    print("Number of Test samples:" + str(len(test_dataset)))

    cam = CAM(nro_rep, dims_list).cuda()
    optimizer = torch.optim.Adam(cam.parameters(), lr=0.001)

    best_cam, best_valid_acc = cam, 0
    for epoch in range(epochs):
        Training_loss, Training_acc = train(train_loader, optimizer, epoch, cam, lista_shapes)
        Validation_loss, Validation_acc = valid(valid_loader, cam, lista_shapes)
        if Validation_acc > best_valid_acc:
            best_valid_acc = Validation_acc
            best_cam = cam
            Testing_loss_best, Testing_acc_best = test(test_loader, cam, lista_shapes)
    return Testing_loss_best, Testing_acc_best

if dataset_target == 'CK+':
    lista_repr_paths = glob.glob(f'../{folder_path_rep}/CK/{nro_rep} REP/{kind_rep}/*')
    y = np.load('../y_ck.npy')
    y = y-1

if dataset_target == 'JAFFE':
    lista_repr_paths = glob.glob(f'../{folder_path_rep}/JAFFE/{nro_rep} REP/{kind_rep}/*')
    y = np.load('../y.npy')

# Load representations
lista_rep = []
for rep_path in lista_repr_paths:
    lista_rep.append(np.load(rep_path))

# No reduce dimensionality to 150
LX = [] 
for i in range(0, len(lista_rep)):
    #pca = PCA(n_components=pca_size) # Just comment PCA elements
    #X = pca.fit(lista_rep[i]).transform(lista_rep[i]) # Just comment PCA elements
    LX.append(lista_rep[i])

final_rep = LX[0]
dims_list = [LX[0].shape[1]]
for i in range(len(lista_rep)-1):
    final_rep = np.hstack((final_rep, LX[i+1]))
    dims_list.append(LX[i+1].shape[1])

#X = np.resize(final_rep, (final_rep.shape[0],final_rep.shape[1], 1))
X = final_rep.copy()
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
    test_loss, test_acc = fit_evaluate_model_CV(X_train, y_train, X_test, y_test, dims_list)
    
    print(f"Attention acc: {test_acc}")
    acc_per_fold.append(test_acc * 100)
    loss_per_fold.append(test_loss)
    

print('------------------------------------------------------------------------')
print(f'Average scores for all folds with CNN Attention nro_rep: {nro_rep}, kind_rep: {kind_rep}, dataset_target: {dataset_target}')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')