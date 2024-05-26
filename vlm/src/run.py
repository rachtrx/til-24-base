import albumentations
from PIL import Image
import IPython.display as display
import torch
import requests
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json

import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
from torchinfo import summary
import urllib
import os

import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar
import torch.nn.functional as F

import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import multiprocessing as mp

NUM_CLASSES=8
TRAIN_NUM_BATCHES = 5982
TEST_NUM_BATCHES = 748
VAL_NUM_BATCHES = 748

# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Enable benchmark mode in cuDNN to find the best algorithm for your hardware
torch.backends.cudnn.benchmark = True
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True

cur_dir = os.getcwd()
vlm_dir = os.path.dirname(cur_dir)
til_dir = os.path.dirname(vlm_dir)
home_dir = os.path.dirname(til_dir)
test_dir = os.path.join(home_dir, 'novice')
img_dir = os.path.join(test_dir, 'images')
metadata_path = os.path.join(test_dir, 'vlm.jsonl')
data_dir = os.path.join(cur_dir, 'data')

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

class BatchDataLoader:
    def __init__(self, batch_path):
        self.batch_path = batch_path

    def load_data(self):
        with open(os.path.join(self.batch_path, "rcnn_img_paths.json"), 'r') as f:
            rcnn_image_paths = json.load(f)
        with open(os.path.join(self.batch_path, "clip_img_paths.json"), 'r') as f:
            clip_image_paths = json.load(f)

        rcnn_image_tensors = self.load_and_stack_images(rcnn_image_paths)
        clip_pixel_values = self.load_and_stack_images(clip_image_paths, img_type='clip')

        with open(os.path.join(self.batch_path, "text_data.json"), 'r') as f:
            text_data = json.load(f)
        text_data_tensors = [self.convert_to_tensors(item) for item in text_data]

        bboxes_batch = self.load_bboxes(os.path.join(self.batch_path, "bboxes.npy"))
        labels_batch = self.load_labels(os.path.join(self.batch_path, "labels.npy"))

        return rcnn_image_tensors, clip_pixel_values, text_data_tensors, bboxes_batch, labels_batch
    
    def load_and_stack_images(self, image_paths, img_type='rcnn'):
        image_tensors = [self.load_image_to_tensor(image_path) for image_path in image_paths]
        image_tensors = [tensor for tensor in image_tensors if tensor is not None]
        if not image_tensors:
            return None
        if img_type == 'rcnn':
            return torch.stack(image_tensors)
        elif img_type == 'clip':
            clip_inputs = [{"pixel_values": tensor.unsqueeze(0)} for tensor in image_tensors]
            return clip_inputs
    
    @staticmethod
    def load_image_to_tensor(image_path):
        image_array = np.load(image_path, mmap_mode='r')
        if image_array is None or image_array.size == 0:
            print(f"Skipping invalid image: {image_path}")
            return None
        # Explicitly copy the array to ensure it's writable
        return torch.from_numpy(image_array.copy()).type(torch.float32)

    @staticmethod
    def convert_to_tensors(data):
        # Ensure each value is converted to a tensor and is copied to be writable
        converted_data = {key: torch.tensor(value).clone() for key, value in data.items()}
        return converted_data

    @staticmethod
    def load_bboxes(bboxes_path):
        bboxes_batch = np.load(bboxes_path, mmap_mode='r')
        # Convert and copy each bounding box to ensure it's writable
        return torch.stack([torch.tensor(b.copy()).view(-1, 4) for b in bboxes_batch])

    @staticmethod
    def load_labels(labels_path):
        labels_batch = np.load(labels_path, mmap_mode='r')
        # Convert and copy each label to ensure it's writable
        return torch.stack([torch.tensor([l]).clone() for l in labels_batch])
        
        
class MemmapIterableDataset(IterableDataset):
    def __init__(self, data, shuffle=False):
        self.type_dir, self.num_batches = data
        self.shuffle = shuffle

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            batch_path = os.path.join(self.type_dir, f"batch_{batch_idx}")
            
            dataloader = BatchDataLoader(batch_path)
            rcnn_image_tensors, clip_pixel_values, text_data_tensors, bboxes_batch, labels_batch = dataloader.load_data()

            # Concatenate the list of tensors along dimension 0 to create a batch
            if rcnn_image_tensors.size(0) == 0 or len(clip_pixel_values) == 0:
                print("No images to process.")
                continue

            yield rcnn_image_tensors, clip_pixel_values, text_data_tensors, bboxes_batch, labels_batch
    

class ImageDetectionModel(pl.LightningModule):
    def __init__(self, train_data, val_data, test_data, num_classes, num_workers):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.rcnn = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.rcnn.roi_heads.box_predictor.cls_score.in_features
        self.rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        self.embedding_transform = nn.Linear(512, 256)
        
        # Allow CLIP model parameters to be trainable (fine-tuning)
        for name, param in self.clip_model.named_parameters():
        # Freeze all parameters first
            param.requires_grad = False

        # Unfreeze parameters in the last layers of the text model
        if 'text_model.encoder.layers.11' in name:
            param.requires_grad = True

        # Unfreeze parameters in the last layers of the vision model
        if 'vision_model.encoder.layers.11' in name:
            param.requires_grad = True

        # Optionally, adjust parameters related to the output projections if fine-tuning the head is desired
        if 'visual_projection.weight' in name or 'text_projection.weight' in name:
            param.requires_grad = True
            
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.train_dataset = MemmapIterableDataset(self.train_data)
        self.val_dataset = MemmapIterableDataset(self.val_data)
        self.test_dataset = MemmapIterableDataset(self.test_data)

    def forward(self, rcnn_imgs_preprocessed, clip_imgs_preprocessed, clip_texts_preprocessed, targets=None):
        batch_feature_maps = []
        # losses =[]

        for rcnn_img, clip_img, clip_text in zip(rcnn_imgs_preprocessed, clip_imgs_preprocessed, clip_texts_preprocessed):
            # Generate embeddings for both image and text from CLIP
            image_embeddings, text_embeddings = self.generate_embeddings(clip_img, clip_text)

            # Ensure image is in [C, H, W] format and transfer to device
            image_tensor = rcnn_img.permute(2, 0, 1).to(self.device)

            # Extract feature maps using the RCNN backbone
            feature_maps = self.get_feature_maps(image_tensor.unsqueeze(0))['0']

            # Modulate feature maps using both image and text embeddings
            modulated_feature_maps = self.modulate_features_with_embeddings(feature_maps, image_embeddings, text_embeddings)
    
            #to remove the batch number
            modulated_feature_maps = modulated_feature_maps.squeeze(0)
            
            # Resize modulated_feature_maps to have size [3, H, W]
            modulated_feature_maps = modulated_feature_maps[:3]  # Take the first 3 channels
            
            # Store processed feature maps
            batch_feature_maps.append(modulated_feature_maps)
            
            # # Compute loss
            # loss = compute_loss(modulated_feature_maps, targeted_feature_maps)
            # losses.append(loss.item())
            
        # print("Dimensions of batch_image_tensors:", [t.shape for t in batch_feature_maps])

        # Since all operations should be on feature maps post backbone processing
        #integrated_features = torch.stack(batch_feature_maps,dim=0)
        
        integrated_features = batch_feature_maps
        
        # print(f"training status: {self.training}")
        
        return self.rcnn(integrated_features, targets)
    
    def generate_embeddings(self, clip_img_preprocessed, clip_text_preprocessed):
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**clip_img_preprocessed).to(self.device)
            text_embeddings = self.clip_model.get_text_features(**clip_text_preprocessed).to(self.device)
        return image_embeddings, text_embeddings
    
    def get_feature_maps(self, image_tensor):
        backbone = self.rcnn.backbone
        backbone.eval()

        with torch.no_grad():
            feature_maps = backbone(image_tensor)

        return feature_maps  # This now returns a dictionary of feature maps
    
    def modulate_features_with_embeddings(self, feature_maps, image_embeddings, text_embeddings):
        # Assuming feature_maps is a batch of feature maps with shape [batch_size, channels, height, width]
        # Both image_embeddings and text_embeddings are [batch_size, 512]
        
        # print(feature_maps.shape)
        
        image_embeddings_transformed = self.embedding_transform(image_embeddings)  # [batch_size, 256]
        text_embeddings_transformed = self.embedding_transform(text_embeddings)    # [batch_size, 256]

        # Expand embeddings to match the spatial dimensions of the feature maps
        image_embeddings_expanded = image_embeddings_transformed.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 256, 1, 1]
        text_embeddings_expanded = text_embeddings_transformed.unsqueeze(-1).unsqueeze(-1)    # [batch_size, 256, 1, 1]
        
        # print(image_embeddings_expanded.shape)

        # Broadcast the embeddings across the spatial dimensions
        image_embeddings_expanded = image_embeddings_expanded.expand_as(feature_maps)  # [batch_size, 256, height, width]
        text_embeddings_expanded = text_embeddings_expanded.expand_as(feature_maps)    # [batch_size, 256, height, width]

        # Concatenate or add embeddings to the feature maps
        # Here we choose concatenation for demonstration; dimension 1 is the channel dimension
        modulated_feature_maps = torch.cat([feature_maps, image_embeddings_expanded, text_embeddings_expanded], dim=1)

        return modulated_feature_maps


    def training_step(self, batch, batch_idx):
        rcnn_image_tensors, clip_image_tensors, clip_text_data, bboxes_batch, labels_batch = batch
        
        # Move tensors to GPU in the training step
        rcnn_image_tensors = rcnn_image_tensors.to(self.device)
        clip_image_tensors = [{key: val.to(self.device) for key, val in item.items()} for item in clip_image_tensors]
        clip_texts_preprocessed = [{key: val.to(self.device) for key, val in item.items()} for item in clip_text_data]
        bboxes_batch = bboxes_batch.to(self.device)
        labels_batch = labels_batch.to(self.device)
        
        targets = []
        for bboxes, labels in zip(bboxes_batch, labels_batch):
            mask = labels != 0
            if mask.any():
                filtered_bboxes = bboxes[mask]
                filtered_labels = labels[mask]

                # Construct the target dictionary
                target = {
                    'boxes': filtered_bboxes.to(self.device),
                    'labels': filtered_labels.to(self.device)
                }
            else:
                # Create an empty target dictionary with correct shape and on the correct device
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float, device=self.device),
                    'labels': torch.zeros(0, dtype=torch.int64, device=self.device)
                }

            targets.append(target)

        outputs = self(rcnn_image_tensors, clip_image_tensors, clip_texts_preprocessed, targets)

        # print("Keys in training")
        # print(outputs.keys())  # Assuming outputs is a dictionary, not a list

        # Calculate total loss from various components
        if any(t['labels'].numel() > 0 for t in targets):
            total_loss = sum(outputs[key] for key in outputs.keys() if 'loss' in key)
            self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return total_loss
        else:
            self.log('train_loss', 0, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return torch.tensor(0.0, requires_grad=True).to(self.device)
    
    def validation_step(self, batch, batch_idx):
        rcnn_image_tensors, clip_image_tensors, clip_text_data, bboxes_batch, labels_batch = batch
        
        # Move tensors to GPU in the training step
        rcnn_image_tensors = rcnn_image_tensors.to(self.device)
        clip_image_tensors = [{key: val.to(self.device) for key, val in item.items()} for item in clip_image_tensors]
        clip_texts_preprocessed = [{key: val.to(self.device) for key, val in item.items()} for item in clip_text_data]
        bboxes_batch = bboxes_batch.to(self.device)
        labels_batch = labels_batch.to(self.device)

        targets = []
        for bboxes, labels in zip(bboxes_batch, labels_batch):
            # Filter out entries where labels are 0 (masking background or padded elements)
            mask = labels != 0
            if mask.any():
                filtered_bboxes = bboxes[mask]
                filtered_labels = labels[mask]

                # Construct the target dictionary
                target = {
                    'boxes': filtered_bboxes.to(self.device),  # Ensure tensors are on the correct device
                    'labels': filtered_labels.to(self.device)
                }
            else:
                # Create an empty target dictionary with correct shape and on the correct device
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float, device=self.device),
                    'labels': torch.zeros(0, dtype=torch.int64, device=self.device)
                }

            targets.append(target)

        self.rcnn.train()

        with torch.no_grad():
            outputs = self(rcnn_image_tensors, clip_image_tensors, clip_texts_preprocessed, targets)

        self.rcnn.eval()

#             print("Keys in validation")
#             print(outputs.keys())

#             print(outputs['loss_classifier'])

        if any(t['labels'].numel() > 0 for t in targets):
            # Calculate the total validation loss by summing individual components
            total_loss = sum(outputs[key] for key in outputs.keys())
            self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return total_loss
        else:
            self.log('val_loss', 0, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return torch.tensor(0.0, device=self.device)
            
    def test_step(self, batch, batch_idx):
        rcnn_image_tensors, clip_image_tensors, clip_text_data, _, _ = batch
        # Assuming test data might not always have labels available

        rcnn_image_tensors = rcnn_image_tensors.to(self.device)
        clip_image_tensors = [{key: val.to(self.device) for key, val in item.items()} for item in clip_image_tensors]
        clip_texts_preprocessed = [{key: val.to(self.device) for key, val in item.items()} for item in clip_text_data]
        
        # Put model in evaluation mode
        self.rcnn.eval()

        # Disable gradient computation explicitly for safety
        with torch.no_grad():
            outputs = self(rcnn_image_tensors, clip_image_tensors, clip_texts_preprocessed)

        # Extract relevant output details, e.g., predicted boxes, labels, and scores
        predictions = {
            'boxes': outputs['boxes'],
            'labels': outputs['labels'],
            'scores': outputs['scores']
        }
        
        return predictions
                
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, persistent_workers=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    checkpoint_path = 'model_checkpoints/vlm_model-epoch=00-val_loss=64.64.ckpt'
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        patience=3,          # no of epochs with no improvement to wait before stopping
        verbose=True,        # logging
        mode='min'           # minimize or maximize the monitored metric
    )

    # Initialize Trainer with model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_checkpoints',
        filename='vlm_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    vlm_model = ImageDetectionModel.load_from_checkpoint(
        checkpoint_path,
        train_data=(train_dir, TRAIN_NUM_BATCHES), 
        val_data=(val_dir, TEST_NUM_BATCHES), 
        test_data=(test_dir, VAL_NUM_BATCHES), 
        num_classes=NUM_CLASSES,
        num_workers=4
    )

    trainer = pl.Trainer(
        max_steps=TRAIN_NUM_BATCHES*49,  # Maximum number of steps (batches) to train for
        callbacks=[checkpoint_callback, early_stopping_callback], # CustomProgressBar()
        val_check_interval=TRAIN_NUM_BATCHES,
        limit_val_batches=VAL_NUM_BATCHES,  # Limit the number of validation batches
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=4
    )
    
    trainer.fit(vlm_model)