import os
import cv2
import math
import copy
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from models.vit import *
from utils import *
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ViTEncoderCIFAR10().to(device)
batch_size = 100
num_workers = 20
num_classes = 10
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
x_train=torch.load("data/svhn/train_x.pt")
y_train=torch.load("data/svhn/train_y.pt")
x_test=torch.load("data/svhn/test_x.pt")
y_test=torch.load("data/svhn/test_y.pt")

trainset = TensorDataset(x_train,y_train) 
train_loader = torch.utils.data.DataLoader(trainset, 
    batch_size=batch_size,shuffle=True,num_workers=num_workers)


testset = TensorDataset(x_test,y_test) 
test_loader = torch.utils.data.DataLoader(testset, 
    batch_size=batch_size,shuffle=True,num_workers=num_workers)

decoder = ViTDecoder(embed_dim=256, patch_size=4, img_size=32).to(device)
decoder.apply(init_weights)
decoder.eval()

PATH = "saved_models/svhn_encoder.pth"
encoder.load_state_dict(torch.load(PATH, weights_only=True))
encoder.eval()

PATH = "saved_models/svhn_decoder.pth"
decoder.load_state_dict(torch.load(PATH, weights_only=True))
decoder.eval()

all_latent_vectors = []
all_labels = []
encoder.eval()
with torch.no_grad():
    for images, labels in train_loader:
        images=images.to(device)
        patch_tokens, cls_feature, _ = encoder(images)
        all_latent_vectors.append(patch_tokens.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        

latent_vectors = np.concatenate(all_latent_vectors, axis=0)
latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
labels_numpy = np.concatenate(all_labels, axis=0)

encoder.eval()
all_latents = []
all_labels = []
count = Counter(y_train.numpy())
desc = [count[i] for i in range(len(count))]

imbal = np.array(desc)
imbal_np = np.array(imbal)
maximum = np.max(imbal_np)
maximum_arg = np.argmax(imbal_np)

x_synth = []
y_synth = []

all_latents_x = []
all_latents_y = []

for index in range(10):
     x1 = latent_vectors[labels_numpy == index]
     y1 = labels_numpy[labels_numpy == index]
     gen_no = imbal[maximum_arg] - imbal[index]
     X_s, Y_s = G_SM1(x1,y1,gen_no,index)
     x_synth.append(X_s)
     y_synth.append(Y_s)
     all_latents_x.append(X_s)
     all_latents_x.append(x1)
     all_latents_y.append(Y_s)
     all_latents_y.append(y1)

x_synth = np.vstack(x_synth)
y_synth = np.hstack(y_synth)

all_latents_x = np.vstack(all_latents_x)
all_latents_y = np.hstack(all_latents_y)

x_synth = torch.tensor(x_synth)
y_synth = torch.tensor(y_synth)

mnist_bal = TensorDataset(x_synth,y_synth) 
num_workers = 20

train_loader = torch.utils.data.DataLoader(mnist_bal, batch_size=batch_size,shuffle=True,num_workers=num_workers)
img_new = []
label_new = []

for x,y in train_loader:
    patch_x = x.view(-1, 64, 256).to(device)
    print(patch_x.shape,"*"*10)
    img = decoder(patch_x.float())
    ximn = img.detach().cpu().numpy()

    img_new.append(ximn)
    label_new.append(y.cpu().numpy())

img_new = np.vstack(img_new)
label_new = np.hstack(label_new)

X_over = np.vstack((img_new,x_train))
Y_over = np.hstack((label_new,y_train))

combx = X_over
comby = Y_over

tensor_x = torch.Tensor(combx)
tensor_y = torch.tensor(comby,dtype=torch.long)
mnist_bal = TensorDataset(tensor_x,tensor_y) 


PATH = "./gen_samples_svhn/"
if os.path.exists(PATH) == False:
    os.mkdir(PATH)
PATH1 = PATH + "./train/"
if os.path.exists(PATH1) == False:
    os.mkdir(PATH1)

ctr = 1
ar = [1 for i in range(10)]
comby = comby.astype("int32")

for i in range(combx.shape[0]):
    img = combx[i]
    img = img / 2 + 0.5     # unnormalize (assuming data is in [-1, 1])
    temp = comby[i]
    
    npimg = np.transpose(img, (1, 2, 0))
    npimg = (npimg * 255).astype(np.uint8)
    
    PATH2 = os.path.join(PATH1, str(temp))
    if not os.path.exists(PATH2):
        os.makedirs(PATH2)  # makedirs is safer than mkdir
    PATH3 = os.path.join(PATH2, '%05d.png' % (ar[temp],))
    cv2.imwrite(PATH3, cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))
    ar[temp] = ar[temp] + 1
    ctr += 1
