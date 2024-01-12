#augmentations??????


import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from datasets import Dataset
from PIL import Image

from torch.utils.data import Dataset
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from transformers import SamModel, SamConfig, SamProcessor
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import skimage
from skimage.morphology import opening, closing, square
from skimage.filters import gaussian



new_path = "/home/phillip-4090/Documents/outputs_256"
db_path = "/home/phillip-4090/Documents/shp_to_tif_output"
patch_size = 256
step = 128

train_dataset_im = os.path.join(new_path, "train", "im")
train_dataset_gt = os.path.join(new_path, "train", "gt")
val_dataset_im = os.path.join(new_path, "val", "im")
val_dataset_gt = os.path.join(new_path, "val", "gt")
if not os.path.exists(train_dataset_im):
    os.makedirs(train_dataset_im)

if not os.path.exists(val_dataset_im):
    os.makedirs(val_dataset_im)

if not os.path.exists(train_dataset_gt):
    os.makedirs(train_dataset_gt)

if not os.path.exists(val_dataset_gt):
    os.makedirs(val_dataset_gt)
def find_files(directory, pattern):
    # List to store paths of matching files
    matching_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for name in files:
            # Check if file matches the pattern
            if pattern in name:
                # Construct full file path and add to the list
                full_path = os.path.join(root, name)
                matching_files.append(full_path)

    return matching_files


Image.MAX_IMAGE_PIXELS = None  # Suppress the warning and remove the limit
db_files = [x.split("/")[-1].split(".tif")[0] for x in find_files(db_path, ".tif") if "mask" not in x]


tifs = []
masks = []
img_dict = {}
mask_dict = {}
for file in db_files:
    print(file)
    valid_indices = []
    # tif = tifffile.imread(os.path.join(train_path, file + ".tif"))
    try:
        with Image.open(os.path.join(db_path, file + ".tif")) as img:
            new_size = (img.width - img.width%patch_size, img.height - img.height%patch_size)
            resized_tif = np.array(img.resize(new_size))
            # print(new_size)
        with Image.open(os.path.join(db_path, "mask_" + file + ".tif")) as mask:
            new_size = (mask.width - mask.width%patch_size, mask.height - mask.height%patch_size)
            resized_mask = np.array(mask.resize(new_size))
            # print(new_size)
        # print(resized_tif.shape)
        # print(resized_mask.shape)
        # print("------------")
        patches_img = patchify(resized_tif, (patch_size, patch_size,3), step=step).squeeze(2)
        patches_mask = patchify(resized_mask, (patch_size, patch_size), step=step)
        print(patches_img.shape)
        print(patches_mask.shape)
        # exit()
        counter = 0
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):

                # exit()
                # print(patches_mask.max())
                # patches_mask[i][j]= (patches_mask[i][j] / 255.).astype(np.uint8)
                # print(patches_mask.max())
                # exit()
                if patches_mask[i][j].max() == 0:
                    print("empty mask")
                    continue
                # print("NOT AN EMPTY MASK")
                patch_mask = patches_mask[i][j]
                # patch_mask[patch_mask > 0] = 255
                patch_of_mask = Image.fromarray(255*patch_mask)
                patch_of_img = Image.fromarray(patches_img[i][j])
                print(train_dataset_gt + "/" + file + "_%s_mask_.png" %str(counter))
                if random.random() < .8:
                    patch_of_mask.save(train_dataset_gt + "/" + file + "_%s.png" %str(counter))
                    patch_of_img.save(train_dataset_im + "/" + file + "_%s.jpg" %str(counter))
                else:
                    patch_of_mask.save(val_dataset_gt + "/" + file + "_%s.png" %str(counter))
                    patch_of_img.save(val_dataset_im + "/" + file + "_%s.jpg" %str(counter))

                counter+=1
    except Exception as e:
        print("bad file??", e)
    # exit()
