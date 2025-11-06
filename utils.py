import os
import torch
import random
import numpy as np
import torch.nn as nn
from pytorch_msssim import ssim
from torchvision.models import vgg16
from sklearn.neighbors import NearestNeighbors
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

def set_seeds(seed_value, use_cuda):
  np.random.seed(seed_value)  # cpu vars
  torch.manual_seed(seed_value)  # cpu  vars
  random.seed(seed_value)  # Python
  os.environ['PYTHONHASHSEED'] = str(seed_value) 
  if use_cuda:
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)  # gpu vars
      torch.backends.cudnn.deterministic = True  # needed
      torch.backends.cudnn.benchmark = False

def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1
    return selected_list

def total_variation_loss(img):
    batch_size = img.size(0)
    h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (h_tv + w_tv) / batch_size

def ssim_loss(x, y):
    return 1 - ssim(x, y, data_range=1.0, size_average=True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def G_SM1(X, y,n_to_sample,cl):

    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    
    return samples, [cl]*n_to_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg16(pretrained=True).features[:9].eval().to(device)
vgg_loss = nn.MSELoss()
def perceptual_loss(x, y):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = (y - mean) / std
    return vgg_loss(vgg(x), vgg(y))


def calculate_metrics(pred, labels,maximum_arg,minimum_arg):
    # Convert probabilities to binary predictions
    binary_preds = torch.argmax(pred, dim=1)
    labels = labels

    # Balanced Accuracy (BACC)
    bacc = balanced_accuracy_score(labels.cpu().numpy(), binary_preds.cpu().numpy())

    # Geometric Mean
    geometric_mean = geometric_mean_score(labels.cpu().numpy(), binary_preds.cpu().numpy(), average = 'macro')

    # F1-score
    gt = labels.cpu().numpy()
    p = binary_preds.cpu().numpy()
    gt = gt.tolist()
    p = p.tolist()

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(gt,p)
    f1 = f1_score(gt, p, average='macro')
    return bacc, geometric_mean, f1, mcc

def train(model, data_loader, optimizer, criterion,maximum_arg,minimum_arg):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs.to(device))
        
        # Calculate loss
        loss = criterion(outputs, labels.to(device))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    
    # Calculate average training loss
    avg_loss = total_loss / total_samples
    
    # Evaluate metrics on the training set
    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            all_preds.append(outputs)
            all_labels.append(labels.to(device))
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate metrics
        bacc, geometric_mean, f1_score, mcc= calculate_metrics(all_preds, all_labels,maximum_arg,minimum_arg)
    
    # Print metrics
    print(f'Training Loss: {avg_loss:.4f} | BACC: {bacc:.4f} | Geometric Mean: {geometric_mean:.4f} | F1-Score: {f1_score:.4f} | MCC: {mcc:.4f}')

    return bacc, geometric_mean, f1_score, mcc

def validate(model, data_loader, criterion,maximum_arg,minimum_arg):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for inputs, labels in data_loader:
            # Forward pass
            outputs = model(inputs.to(device))
            
            # Calculate loss
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            all_preds.append(outputs)
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        bacc, geometric_mean, f1_score, mcc = calculate_metrics(all_preds, all_labels,maximum_arg,minimum_arg)
    
    # Calculate average validation loss
    avg_loss = total_loss / total_samples
    
    # Print validation metrics
    print(f'Validation Loss: {avg_loss:.4f} | BACC: {bacc:.4f} | Geometric Mean: {geometric_mean:.4f} | F1-Score: {f1_score:.4f} | MCC: {mcc:.4f}')

    return bacc, geometric_mean, f1_score, mcc