import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision import transforms
import time

'''
This script reimplements DynamicDet, a dynamic object detection 
model that adapts its architecture based on input image complexity. 
'''

class Router(nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1) # Predicted score
        )

    def forward(self, features):
        '''
        features: dict or list of tensors from FPN
        We want to flatten the average pooled features
        '''
        pooled = []

        if isinstance(features, dict):
            features = list(features.values())
        
        # First pool the features
        for feat in features:
            p = self.pool(feat).view(feat.size(0), -1) # (B, C)
            pooled.append(p)

        # Concatenate features
        concat = torch.stack(pooled, dim=0).mean(dim=0) # (B, C)

        # Pass through Fully Connected layers
        output = self.fc(concat) # (B, 1)
        score = torch.sigmoid(output).squeeze(-1) # (B,)

        return score
    
def replace_box_predictor(model, num_classes):
    '''
    Replace the box predictor to match the number of classes
    '''
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
    
'''
This is the DynamicDet class with the 2 detectors.
The lightweight detector (B1) will be the Faster RCNN 
MobileNet V3, and the heavy detector (B2) will be the
Faster RCNN ResNet50.
'''
class DynamicDet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Lightweight backbone (B1)
        self.b1 = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.b1 = replace_box_predictor(self.b1, num_classes)

        # Heavy backbone (B2)
        self.b2 = fasterrcnn_resnet50_fpn(pretrained=True)
        self.b2 = replace_box_predictor(self.b2, num_classes)

        # Router
        self.router = Router(in_channels=256, hidden=256).to(device)
        self.threshold = 0.5 # Initial threshold for router

    def get_first_backbone_features(self, images):
        return self.b1.backbone(images)
    
    def forward(self, images, targets, train_router):
        '''
        Params:
            images: list[Tensor]
            targets: list[dict]
            train_routers: Boolean - Compute B1 & B2
        '''
        if self.training:
            # Run backbones
            losses_b1 = self.b1(images, targets)
            losses_b2 = self.b2(images, targets)

            loss_b1 = sum(losses_b1.values())
            loss_b2 = sum(losses_b2.values())

            outputs = {
                "Loss_B1": loss_b1,
                "Loss_B2": loss_b2
            }

            if train_router:
                # Train router with B1 & B2 and evaluate loss
                with torch.no_grad():
                    per_loss_b1 = []
                    per_loss_b2 = []
                    for image, target in zip(images, targets):
                        l1 = sum(v for v in self.b1([image], [target]).values())
                        l2 = sum(v for v in self.b2([image], [target]).values())

                        per_loss_b1.append(l1.detach().cpu())
                        per_loss_b2.append(l2.detach().cpu())
                    
                per_loss_b1 = torch.tensor(per_loss_b1, device=device)
                per_loss_b2 = torch.tensor(per_loss_b2, device=device)

                # 1 means go to B2, 0 means exit
                margin = 0.1
                routing_label = (per_loss_b2 + margin < per_loss_b1).float().to(device)

                features = self.get_first_backbone_features(images)
                score = self.router(features)

                # BCE loss
                bce = F.binary_cross_entropy(score, routing_label)

                outputs["Loss_Router"] = bce
                outputs["Router"] = routing_label
                outputs["Router_Score"] = score.detach()

                return outputs
            else:
                # Run B1 and return
                features = self.get_first_backbone_features(images)
                score = self.router(features)
                detections = []

                for i, image in enumerate(images):
                    score_i = score[i].item()
                    if score_i < self.threshold:
                        # Run B1 for image
                        detection = self.b1([image])
                        detections.append(detection[0])
                    else:
                        # Run B2 for image
                        detection = self.b2([image])
                        detections.append(detection[0])
                
                return detections


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")