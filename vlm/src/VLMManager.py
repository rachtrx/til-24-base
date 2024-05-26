from typing import List
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from PIL import Image
import base64
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torch.utils.data import DataLoader

class VLMManager:
    def __init__(self):
        # initialize the model here
        self.rcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.rcnn_preprocessor = self.rcnn_weights.transforms()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def decode_and_preprocess_image(self, image_bytes):
        # Decode the image bytes
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Preprocess the image using the CLIP processor
        clip_image = self.clip_processor(images=image, return_tensors="pt")
        # Prepare the image for RCNN
        rcnn_image = self.rcnn_preprocessor(image)
        return clip_image, rcnn_image
    
    def preprocess_text(self, text):
        clip_text = self.clip_processor(text=text, return_tensors="pt")
        return clip_text

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        ## image and caption pass it into the model
        
        ## preprocess the stuff
        clip_image, rcnn_image = self.decode_and_preprocess_image(image)
        clip_text = self.preprocess_text(caption)
        # Move data to the correct device
        clip_image = {k: v.to(self.model.device) for k, v in clip_image.items()}
        clip_text = {k: v.to(self.model.device) for k, v in clip_text.items()}
        rcnn_image = rcnn_image.to(self.model.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model([rcnn_image], [clip_image], [clip_text]) # targets is none, so list of [{'boxes': [], 'labels': [], 'scores': []}]
        
        ## return the box
        print(predictions)
        if predictions and len(predictions) > 0 and predictions[0].get('boxes') is not None and predictions[0]['boxes'].numel() > 0:
            box = predictions[0]['boxes'].cpu().numpy().tolist()
            box = self.convert_bbox_to_dimensions(box)
        else:
            box = [0, 0, 0, 0]
        return box
    
    @staticmethod
    def convert_bbox_to_dimensions(bbox): # CONVERT
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        width = x2 - x1
        height = y2 - y1
        return [x, y, width, height]


class ImageDetectionModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.rcnn = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = self.rcnn.roi_heads.box_predictor.cls_score.in_features
        self.rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        self.embedding_transform = nn.Linear(512, 256)

    def forward(self, rcnn_imgs_preprocessed, clip_imgs_preprocessed, clip_texts_preprocessed, targets=None):
        batch_feature_maps = []
        # losses =[]

        for rcnn_img, clip_img, clip_text in zip(rcnn_imgs_preprocessed, clip_imgs_preprocessed, clip_texts_preprocessed):
            # Generate embeddings for both image and text from CLIP
            image_embeddings, text_embeddings = self.generate_embeddings(clip_img, clip_text)

            # Ensure image is in [C, H, W] format and transfer to device
            # image_tensor = rcnn_img.permute(2, 0, 1).to(self.device) # IMPT no longer needed, not saved as numpy anymore after preprocessing

            # Extract feature maps using the RCNN backbone
            feature_maps = self.get_feature_maps(rcnn_img.unsqueeze(0))['0']

            # Modulate feature maps using both image and text embeddings
            modulated_feature_maps = self.modulate_features_with_embeddings(feature_maps, image_embeddings, text_embeddings)
    
            #to remove the batch number
            modulated_feature_maps = modulated_feature_maps.squeeze(0)
            
            # Resize modulated_feature_maps to have size [3, H, W]
            modulated_feature_maps = modulated_feature_maps[:3]  # Take the first 3 channels
            
            # Store processed feature maps
            batch_feature_maps.append(modulated_feature_maps)

        integrated_features = torch.stack(batch_feature_maps, dim=0)
        
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
        
        image_embeddings_transformed = self.embedding_transform(image_embeddings)  # [batch_size, 256]
        text_embeddings_transformed = self.embedding_transform(text_embeddings)    # [batch_size, 256]

        # Expand embeddings to match the spatial dimensions of the feature maps
        image_embeddings_expanded = image_embeddings_transformed.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 256, 1, 1]
        text_embeddings_expanded = text_embeddings_transformed.unsqueeze(-1).unsqueeze(-1)    # [batch_size, 256, 1, 1]

        # Broadcast the embeddings across the spatial dimensions
        image_embeddings_expanded = image_embeddings_expanded.expand_as(feature_maps)  # [batch_size, 256, height, width]
        text_embeddings_expanded = text_embeddings_expanded.expand_as(feature_maps)    # [batch_size, 256, height, width]

        # Concatenate or add embeddings to the feature maps
        # Here we choose concatenation for demonstration; dimension 1 is the channel dimension
        modulated_feature_maps = torch.cat([feature_maps, image_embeddings_expanded, text_embeddings_expanded], dim=1)

        return modulated_feature_maps