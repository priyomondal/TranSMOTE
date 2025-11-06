import os
import torch
import random
import models
import sklearn
import imblearn
import argparse
import numpy as np
from utils import *
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.metrics import f1_score
from torchvision import datasets, transforms
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--state', default=42, type=int,
                    help='seed for initializing training. ')

args1 = parser.parse_args()
set_seeds(args1.state, True)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 32

data_transforms = {
    'train': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
}
import time
t0 = time.time()

data_dir = ["gen_samples_svhn"]
for i in range(len(data_dir)):

    print("\n\n dataset:",data_dir[i],"\n\n")

    train_dataset = datasets.ImageFolder(os.path.join(data_dir[i], 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join("data", 'svhn_val_cv2'), data_transforms['val'])


    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
    val_dataloder = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=16)

    model = models.__dict__['resnet32'](num_classes=10, use_norm=False).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_val_bacc = 0.0  # Initialize the best validation BACC
    best_epoch = 0

    # Assuming you have already defined your model, train_loader, val_loader, criterion, and optimizer
    num_epochs = 200  # Change this to the desired number of epochs
    best_validation_acc = 0.0
    best_gmean = 0.0
    best_f1 = 0.0
    best_mcc = 0.0

    f = open("./"+"metrics_resnet32_"+str(args1.state)+".txt","a")
    f.write("\n"+"metrics of SVHN:"+data_dir[i]+"\n")
    f.write("bal_acc,mathews,f1,geometric")

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        dec_y = torch.load("data/svhn/train_y.pt").type(torch.int64)
        count = Counter(dec_y.numpy())
        desc = [count[i] for i in range(len(count))]
        imbal_np = np.array(desc)
        maximum_arg = np.argmax(imbal_np)
        minimum_arg = np.argmin(imbal_np)
        
        # Train the model
        bacc, geometric_mean, f1, mcc= train(model, train_dataloder, optimizer, criterion,maximum_arg,minimum_arg)
        
        # Validate the model
        bacc, geometric_mean, f1, mcc= validate(model, val_dataloder, criterion,maximum_arg,minimum_arg)

        # Update best validation accuracy and save the model if it improved
        if bacc > best_validation_acc:
            best_validation_acc = bacc
            best_gmean =  geometric_mean
            best_f1 =  f1
            best_mcc =  mcc

            torch.save(model.state_dict(), data_dir[i]+'.pth')  # Change the filename as needed
            print(f'Saved model with improved validation accuracy: {best_validation_acc:.4f}')
        print(f'Best validation accuracy: {best_validation_acc:.4f}')

        t1 = time.time()
        print('total time(min): {:.2f}'.format((t1 - t0)/60))             
 
    f.write("\n"+str(best_validation_acc)+","+str(best_mcc)+","+str(best_f1)+","+str(best_gmean))
    f.close()
    print('Training complete.')
