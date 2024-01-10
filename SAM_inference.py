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



train_path = "/home/phillip/Documents/35803/outputs3"
#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox


class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_mito_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_mito_model.load_state_dict(torch.load("./data/mito_model_checkpoint.pth"))

device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_model.to(device)


# Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if x_indices.size > 0 and y_indices.size > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    return -1


# Apply a trained model on large image
large_test_images = tifffile.imread("./data/training.tif")
large_test_mask = tifffile.imread("./data/training_groundtruth.tif")[1]
large_test_image = large_test_images[1]
print(large_test_image.shape)
print(large_test_mask.shape)
print("---------------------------")
patches = patchify(large_test_image, (256, 256), step=128)  # Step=256 for 256 patches means no overlap
mask_patches = patchify(large_test_mask, (256, 256), step=128)
patches.shape
print(patches.shape)
my_mito_model.eval()
# processed_patches = np.empty(patches.shape[:-2] + (256,256))
processed_patches = np.empty(patches.shape)
processed_patches2 = np.empty(patches.shape)
processed_patches3 = np.empty(patches.shape)
processed_patches4 = np.empty(patches.shape)
print(processed_patches.shape)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = Image.fromarray(patches[i][j])
        prompt = get_bounding_box(torch.tensor(mask_patches[i][j]))
        if prompt == -1:
            processed_patches[i][j] = np.empty(patch.size)
            print("negative prompt")
        else:
            inputs = processor(patch, input_boxes=[[prompt]], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                # print(my_mito_model(**inputs, multimask_output=False))

                seg_prob = my_mito_model(**inputs, multimask_output=False).pred_masks.squeeze(1)
                seg_prob = seg_prob.cpu().numpy().squeeze()
                mask = (seg_prob > 0.9).astype(np.uint8)

                # Smoothing the mask with scikit-image
                # Apply Gaussian Blur
                blurred_mask = gaussian(mask, sigma=3)

                # Apply Morphological Operations (choose one as needed)
                # Define a square structuring element for morphological operations
                selem = square(5)

                mask_opened = opening(blurred_mask, selem)
                mask_closed = closing(blurred_mask, selem)

                # Choose which mask to use for further processing or storing
                # final_mask = mask_closed  # or mask_opened or blurred_mask
                final_mask = mask_opened
                # Store the processed patch
                processed_patches[i][j] = mask
                processed_patches2[i][j] = mask_opened
                processed_patches3[i][j] = mask_closed
                processed_patches4[i][j] = blurred_mask

reconstructed_image = unpatchify(processed_patches, large_test_image.shape)
reconstructed_image2 = unpatchify(processed_patches2, large_test_image.shape)
reconstructed_image3 = unpatchify(processed_patches3, large_test_image.shape)
reconstructed_image4 = unpatchify(processed_patches4, large_test_image.shape)

fig, axs = plt.subplots(nrows=2, ncols=2) # This creates a 2x2 grid of subplots

axs[0,0].imshow(reconstructed_image)
axs[0,1].imshow(reconstructed_image2)
axs[1,0].imshow(reconstructed_image3)
axs[1,1].imshow(reconstructed_image4)
plt.show()
#
# plt.imshow(large_test_mask)
# plt.show()
#
# original_image = unpatchify(patches, large_test_image.shape)
# plt.imshow(original_image)
# plt.show()


