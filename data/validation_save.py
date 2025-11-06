import torch
import os
print(torch.version.cuda) 
import numpy as np
import cv2
# import time
from torchvision.utils import save_image


combx = torch.load("test_x.pt").numpy()
comby = torch.load("test_y.pt").numpy()


PATH = "./svhn_val_cv2/"
if os.path.exists(PATH) == False:
    os.mkdir(PATH)

ctr = 1
from collections import Counter

ar = [1 for i in range(10)]
comby = comby.astype("int32")

for i in range(combx.shape[0]):
    img = combx[i]
    # img = torch.tensor(img)

    img = img / 2 + 0.5     

    # img = torch.tensor(img)
    npimg = np.transpose(img, (1, 2, 0))
    # npimg = npimg[:, :, ::-1] #####added
    npimg = (npimg * 255).astype(np.uint8)

    temp = comby[i]

    PATH1 = os.path.join(PATH+str(temp)+"/")
    if os.path.exists(PATH1) == False:
        os.mkdir(PATH1)


    PATH2 = os.path.join(PATH1, '%05d.png' % (ar[temp],))
    cv2.imwrite(PATH2, cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))
    # save_image(img.clamp(0,1), PATH2)

    ar[temp] = ar[temp] + 1

    ctr += 1


# ctr = 1
# ar = [1 for i in range(10)]
# comby = comby.astype("int32")

# for i in range(combx.shape[0]):
#     img = combx[i]
#     img = img / 2 + 0.5     # unnormalize (assuming data is in [-1, 1])
#     temp = comby[i]
    
#     # img = torch.tensor(img)
#     npimg = np.transpose(img, (1, 2, 0))
#     # npimg = npimg[:, :, ::-1] #####added
#     npimg = (npimg * 255).astype(np.uint8)
    
#     PATH2 = os.path.join(PATH1, str(temp))
#     if not os.path.exists(PATH2):
#         os.makedirs(PATH2)  # makedirs is safer than mkdir
#     PATH3 = os.path.join(PATH2, '%05d.png' % (ar[temp],))
#     # save_image(img.clamp(0,1), PATH3)
#     cv2.imwrite(PATH3, cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))
#     ar[temp] = ar[temp] + 1
#     ctr += 1
