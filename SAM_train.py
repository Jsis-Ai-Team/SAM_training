#augmentations??????

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from torchvision.io import read_image
import cv2
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from datasets import Dataset
from PIL import Image, ImageOps

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.optim import Adam, AdamW
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


#Training loop
num_epochs = 1000
validation_step = 100000
train_path = "/home/phillip/Documents/35803/outputs_3"
save_img_path = "./runs/saved_images"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)

patch_size = 512
step = 256
#Get bounding boxes from mask.
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
        # print(bbox)

        return bbox
    print("THIS SHOULD NOT FUCKING HAPPEN")
    return [1,1,3,3]

from skimage import measure
def custom_collate_fn(batch):
    # Extract separate lists
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]

    # Stack images as they should have the same size
    images = torch.stack(images, dim=0)

    # Handle boxes separately (e.g., by padding, or leave them as lists)

    # Combine into a new batch
    batch = {'images': images, 'boxes': boxes}
    return batch

def get_multiple_bounding_boxes(ground_truth_map):
    # Label each separate object in the mask
    labels = measure.label(ground_truth_map > 0)
    bboxes = []

    # Iterate over each detected object
    for region in measure.regionprops(labels):
        # Get coordinates
        minr, minc, maxr, maxc = region.bbox

        # You can adjust coordinates here as needed
        bbox = [minc, minr, maxc, maxr]
        bboxes.append(bbox)

    return bboxes


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



class SAMDataset(Dataset):
    def __init__(self, images, masks, processor):
        """
        Args:
            images_dir (string): Path to all images.
            masks_dir (string): Path to all masks.
            processor (callable): Processor function to preprocess the images.
        """
        self.processor = processor
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # try:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
            # Load image and mask
            image = Image.open(image_path)
            # image = Image.open(image_path).resize((256,256))
            mask = Image.open(mask_path)
            # mask = Image.open(mask_path).resize((256,256))
            image = np.array(image)
            # image = read_image(image_path).resize(3, 256,256)
            # mask = read_image(mask_path).resize(1,256,256)
            ground_truth_mask = np.array(mask).squeeze()

            boxed_masks = []
            # prompt = get_bounding_box(ground_truth_mask)
            prompt = get_multiple_bounding_boxes(ground_truth_mask)

            # prepare image and prompt for the model
            for box in prompt:
                minc, minr, maxc, maxr = box
                # Create a copy of the image to modify
                modified_mask = np.zeros(ground_truth_mask.shape)
                modified_mask[minr:maxr, minc:maxc] = ground_truth_mask[minr:maxr, minc:maxc]


                # Set pixels outside the bbox to black (0)
                # modified_mask[~mask] = 0
                boxed_masks.append(modified_mask)
            boxed_masks = np.array(boxed_masks)
            inputs = self.processor(image, input_boxes=[prompt], return_tensors="pt")
            # remove batch dimension which the processor adds by default
            # add ground truth segmentation
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = boxed_masks
            inputs["image"] = image


            return inputs
        # except Exception as e:
        #     print(f"skipping broken image: {image_path}, {mask_path}: {e}")
        #     idx = (idx + 1)%len(self)
        #     return self.__getitem__(idx)

imgs = []
masks = []
train_dataset = os.path.join(train_path, "train")
val_dataset = os.path.join(train_path, "val")


train_imgs = find_files(train_dataset, "_img_")#[:int(len(train_dataset)/10)]
random.shuffle(train_imgs)
train_imgs = train_imgs[:int(len(train_imgs)/10)]
train_masks = [x.replace("_img_", "_mask_") for x in train_imgs]
val_imgs = find_files(val_dataset, "_img_")#[:int(len(val_dataset)/10)]
random.shuffle(val_imgs)
val_imgs = val_imgs[:int(len(val_imgs)/10)]
val_masks = [x.replace("_img_", "_mask_") for x in val_imgs]

print(len(train_imgs))
print(len(val_imgs))
# exit()
# Create the dataset using the datasets.Dataset class


processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# print(processor)
# Create an instance of the SAMDataset
sam_train_dataset = SAMDataset(train_imgs, train_masks, processor=processor)
sam_val_dataset = SAMDataset(val_imgs, val_masks, processor=processor)
# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(sam_train_dataset, batch_size=1, shuffle=True, drop_last=False)
val_dataloader = DataLoader(sam_val_dataset, batch_size=1, shuffle=True, drop_last=False)
# batch = next(iter(train_dataloader))


# Load the model
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
model.load_state_dict(torch.load("./models/model_epoch_6_batch_idx_0.pth"))
#
# model = SamModel.from_pretrained("facebook/sam-vit-base")
# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)


# Initialize the optimizer and the loss function
# optimizer = AdamW(model.mask_decoder.parameters(), lr=1e-6, weight_decay=1e-4)
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# TensorBoard writer
writer = SummaryWriter()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

model.train()
loss=0
for epoch in range(num_epochs):
    epoch_losses = []
    model.train()
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        # forward pass

        outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)
        # compute loss
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        predicted_masks = outputs.pred_masks.squeeze(2)
        loss = seg_loss(predicted_masks, ground_truth_masks)
        optimizer.zero_grad()
        loss.backward()
        # optimize

        if batch_idx%100 == 0:
            n_boxes = predicted_masks.shape[1]
            save_pred_0 = predicted_masks[0]
            save_mask_0 = ground_truth_masks[0]
            final_save_arr = None
            for i in range(n_boxes):
                predicted_masks = 255*torch.sigmoid(save_pred_0[i]).detach().cpu()
                ground_truth_masks = 255*save_mask_0[i].detach().cpu()

                final_save = torch.hstack((ground_truth_masks, predicted_masks)).squeeze()
                # print(final_save.shape)
                if final_save_arr is None:
                    final_save_arr = final_save
                else:
                    final_save_arr = torch.vstack((final_save_arr, final_save))

            final_image = Image.fromarray(final_save_arr.numpy()).convert('L')
            final_image.save(save_img_path +"/multiple_bbox_epoch_tenth" + str(epoch) + "_"+ str(batch_idx) +".png")
            optimizer.step()
            epoch_losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + batch_idx)

        if batch_idx % validation_step == 0 and epoch !=0:
            with torch.no_grad():
                model.eval()
                val_losses = []
                for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                # for batch in val_dataloader:
                    # Load validation data to the device
                    pixel_values = batch["pixel_values"].to(device)
                    input_boxes = batch["input_boxes"].to(device)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)

                    # Forward pass
                    outputs = model(pixel_values=pixel_values,
                                    input_boxes=input_boxes,
                                    multimask_output=False)

                    # Compute loss
                    predicted_masks = outputs.pred_masks.squeeze(2)
                    val_loss = seg_loss(predicted_masks, ground_truth_masks)
                    # Accumulate the validation loss
                    val_losses.append(val_loss.item())

                    # Compute the average validation loss
                    avg_val_loss = np.mean(val_losses)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
            model.train()
            print(f'batch_idx: {batch_idx}')
            print(f'Mean loss: {mean(epoch_losses)}')
            torch.save(model.state_dict(), f'./models/multiple_bbox_epoch{epoch}.pth')
# Print epoch statistics
            print(f'batch_idx: {batch_idx}, Train Loss: {np.mean(epoch_losses)}, Val Loss: {np.mean(val_losses)}')
    # torch.save(model.state_dict(), f"./data/checkpoint_full_epoch_{epoch}.pth")
writer.close()
# Save the model's state dictionary to a file
