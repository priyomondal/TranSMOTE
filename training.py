import os
import cv2
import math
import copy
import torch
import numpy as np
import torchvision
import setproctitle
from utils import *
import torch.nn as nn
from models.vit import *
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from collections import Counter
from torchvision import transforms
from torchvision.models import vgg16
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.vision_transformer import VisionTransformer
setproctitle.setproctitle("/usr/lib/xorg/Xorg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 100
num_workers = 20
num_epochs = 200
x_train=torch.load("data/svhn/train_x.pt")
y_train=torch.load("data/svhn/train_y.pt")
x_test=torch.load("data/svhn/test_x.pt")
y_test=torch.load("data/svhn/test_y.pt")

encoder = ViTEncoder().to(device)
trainset = TensorDataset(x_train,y_train) 
train_loader = torch.utils.data.DataLoader(trainset, 
    batch_size=batch_size,shuffle=True,num_workers=num_workers)

testset = TensorDataset(x_test,y_test) 
test_loader = torch.utils.data.DataLoader(testset, 
    batch_size=batch_size,shuffle=True,num_workers=num_workers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

decoder = ViTDecoder(embed_dim=256, patch_size=4, img_size=32).to(device)

decoder.apply(init_weights)
decoder.eval()
l1_loss=nn.SmoothL1Loss()

vgg = vgg16(pretrained=True).features[:9].eval().to(device)
for p in vgg.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-5)
decoder.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)

        with torch.no_grad():
             patch_tokens, _, _ = encoder(images)
          
        recon_images = decoder(patch_tokens)

        loss = (
            l1_loss(recon_images, images)
            + 0.1 * perceptual_loss(recon_images, images)
            + 0.01 * total_variation_loss(recon_images)
            + 0.5 * ssim_loss(recon_images, images))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Decoder Loss: {avg_loss:.4f}")

torch.save(encoder.state_dict(), "saved_models/svhn_encoder.pth")
torch.save(decoder.state_dict(), "saved_models/svhn_decoder.pth")
