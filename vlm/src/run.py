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

import multiprocessing as mp

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

class MemmapIterableDataset(IterableDataset):
    def __init__(self, data, shuffle=False):
        self.type_dir, self.num_batches = data
        self.shuffle = shuffle

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            batch_path = os.path.join(self.type_dir, f"batch_{batch_idx}")


            # Load file paths to images from JSON file
            image_paths_file = os.path.join(batch_path, "img_paths.json")
            with open(image_paths_file, 'r') as f:
                image_paths_list = json.load(f)

            # Unpack the list of image paths
            image_paths = [image_path for image_path in image_paths_list]

            # Load other batch data
            bboxes_path = os.path.join(batch_path, "bboxes.npy")
            labels_path = os.path.join(batch_path, "labels.npy")
            text_features_path = os.path.join(batch_path, "text_features.npy")

            # Load other batch data as numpy arrays
            # try:
            bboxes_batch = np.load(bboxes_path)
            labels_batch = np.load(labels_path)
            # except ValueError:
            #     # If there's an error suggesting that you need to allow pickling, use allow_pickle=True
            #     bboxes_batch = np.load(bboxes_path, allow_pickle=True)
            #     labels_batch = np.load(labels_path, allow_pickle=True)
            text_features_batch = np.load(text_features_path, mmap_mode='r')

            # Convert numpy arrays to torch tensors
            bboxes_batch = torch.stack([torch.tensor(b).view(-1, 4) for b in bboxes_batch])
            labels_batch = torch.stack([torch.tensor([l]) for l in labels_batch])
            # print(bboxes_batch)
            # print(labels_batch)
            text_features_batch = torch.tensor(text_features_batch) # TO CHECK IF DIM IS TOO HIGH
            # print(text_features_batch)

            image_tensors = []
            for image_path in image_paths:
                # Load the image data as a memory-mapped array
                image_array = np.load(image_path, mmap_mode='r')

                if image_array is None or image_array.shape[0] == 0 or image_array.shape[1] == 0:
                    print(f"Skipping invalid image: {image_path}")
                    continue

                # Convert to a PyTorch tensor
                image_tensor = torch.tensor(image_array, dtype=torch.float32)

                # Transfer to GPU
                image_tensor = image_tensor.to('cuda')

                # Append the tensor to the list
                image_tensors.append(image_tensor)

            # Now concatenate the list of tensors along dimension 0 to create a batch
            if not image_tensors:
                print("No images to process.")
                continue
                
            # bboxes_batch = bboxes_batch.to('cpu')
            # labels_batch = labels_batch.to('cpu')
            # text_features_batch = text_features_batch.to('cpu')

            yield image_tensors, bboxes_batch, labels_batch, text_features_batch
            
class ObjectDetectionModule(pl.LightningModule):
    def __init__(self, num_classes, train_data, val_data, test_data, learning_rate=1e-3, num_workers=4):
        super().__init__()
        # Account for an additional dummy class for padding
        self.num_classes = num_classes + 1  # Increase number of classes to include a dummy class
        # Initialize the Faster R-CNN and CLIP models
        self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Replace the classifier in Faster R-CNN
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        clip_out_dim = 512  # Adjust according to your specific model output
        self.projection = torch.nn.Linear(clip_out_dim, in_features)

        # Define the fusion layer
        # self.fusion_layer = FusionLayer(visual_feature_dim=in_features, text_feature_dim=512, output_dim=num_classes)
        self.learning_rate = learning_rate
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.train_dataset = MemmapIterableDataset(self.train_data, shuffle=True) # need transform?
        self.val_dataset = MemmapIterableDataset(self.val_data)
        self.test_dataset = MemmapIterableDataset(self.test_data)

    def forward(self, images, text_features=None, targets=None):
        if targets is not None:
            # During training, we expect a dictionary of losses
            outputs = self.faster_rcnn(images, targets=targets)
            if self.training and text_features is not None:
                outputs = self.augment_losses_with_text(outputs, targets, text_features)
        else:
            # During validation without targets, we are in inference mode
            outputs = self.faster_rcnn(images)  # This will return a list of detections
        return outputs
    
    def augment_losses_with_text(self, outputs, targets, text_features):
        # Directly modify the outputs dictionary
        loss_classifier = outputs['loss_classifier']
        loss_box_reg = outputs['loss_box_reg']
        
        text_features_proj = self.projection(text_features)

        # Adjust the classifier loss using text features
        for i, target in enumerate(targets):
            target_labels = target['labels']
            text_sim = torch.cosine_similarity(text_features_proj[i].unsqueeze(0),
                                               self.faster_rcnn.roi_heads.box_predictor.cls_score.weight.data[target_labels], dim=1)
            loss_classifier += (1.0 - text_sim).mean()  # Adding a penalty term based on text similarity

        # Update the losses in the outputs dictionary
        outputs['loss_classifier'] = loss_classifier
        outputs['loss_box_reg'] = loss_box_reg
        return outputs
    
    def training_step(self, batch, batch_idx):
        images, bboxes, labels, text_features = batch
        # DEBUG
        batch_size = len(bboxes[0])
        
        targets = [{'boxes': bbox, 'labels': label} for bbox, label in zip(bboxes, labels)]
        loss_dict = self(images, text_features, targets=targets)
        self.log('train_loss', sum(loss_dict.values()))
        return sum(loss_dict.values())
    
    def validation_step(self, batch, batch_idx):
        images, bboxes, labels, text_features = batch
        targets = [{'boxes': bbox, 'labels': label} for bbox, label in zip(bboxes, labels)]
        outputs = self(images, text_features, targets=targets)  # This should return raw predictions, not losses
    
        losses = []
        for output, target in zip(outputs, targets):
            if 'scores' in output:
                max_conf_index = output['scores'].argmax()  # Get the index of the highest score
                pred_box = output['boxes'][max_conf_index].unsqueeze(0)  # Select the box and unsqueeze

                true_box = target['boxes']
                loss = F.mse_loss(pred_box, true_box)  # Compute the loss
                losses.append(loss)

        total_loss = torch.mean(torch.stack(losses))  # Average the losses
        self.log('val_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers)



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    data_module = ObjectDetectionModule(
        num_classes=8,
        train_data=(train_dir, 1709),
        val_data=(val_dir, 214),
        test_data=(test_dir, 214),
        learning_rate=1e-3,
        num_workers=0,
    )

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
        filename='asr_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_steps=1709*10,  # Maximum number of steps (batches) to train for
        callbacks=[checkpoint_callback, early_stopping_callback], # CustomProgressBar()
        val_check_interval=1709,
        limit_val_batches=214,  # Limit the number of validation batches
    )
    
    trainer.fit(data_module)