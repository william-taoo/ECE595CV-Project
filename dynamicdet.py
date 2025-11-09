import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn
)
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
from handling_data import (
    process_dataset, 
    get_annotation_info, 
    split_dataset,
    COCO_CLASSES
)

'''
This script reimplements DynamicDet, a dynamic object detection 
model that adapts its architecture based on input image complexity. 
'''

def replace_box_predictor(model, num_classes):
    '''
    Replace the box predictor to match the number of classes
    '''
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1) # +1 for background
    return model

def get_transform():
    # Conver to tensor
    t = []
    t.append(T.ToTensor())
    return T.Compose(t)

def collate_fn(batch):
    return tuple(zip(*batch))

def sanitize_boxes(targets):
    for t in targets:
        boxes = t["boxes"]
        if len(boxes) == 0:
            continue

        # Ensure all boxes are in [xmin, ymin, xmax, ymax] order
        xmin = torch.min(boxes[:, 0], boxes[:, 2])
        ymin = torch.min(boxes[:, 1], boxes[:, 3])
        xmax = torch.max(boxes[:, 0], boxes[:, 2])
        ymax = torch.max(boxes[:, 1], boxes[:, 3])
        t["boxes"] = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    return targets

class CocoDataset(Dataset):
    def __init__(self, root, annotation, transform):
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.transform = transform

        # Make contiguous category mapping here
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        self.id_to_contiguous = {cat['id']: i + 1 for i, cat in enumerate(cats)}
        self.contiguous_to_name = {i + 1: cat['name'] for i, cat in enumerate(cats)}

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for obj in anns:
            xmin, ymin, w, h = obj['bbox']
            if w <= 0 or h <= 0: # Skip invalid boxes
                continue
            boxes.append([xmin, ymin, xmin + w, ymin + h])

            # Remap categories
            labels.append(self.id_to_contiguous[obj['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

class Router(nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1) # Predicted Score
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, in_channels]
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.view(-1, 1) # Output shape [B, 1]
    
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
        self.b1 = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        self.b1 = replace_box_predictor(self.b1, num_classes)

        # Heavy backbone (B2)
        self.b2 = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.b2 = replace_box_predictor(self.b2, num_classes)

        # Determine feature size from B1 backbone
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.b1.backbone(dummy_input)
            concat_feats = []
            for feat_map in feats.values():
                pooled = self.adaptive_pool(feat_map)
                concat_feats.append(pooled.squeeze())
            combined = torch.cat(concat_feats, dim=0)
            feature_size = combined.shape[0]
        
        print(f"Router input feature size: {feature_size}")

        # Router
        self.router = Router(in_channels=feature_size, hidden=256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.threshold = 0.5 # Initial threshold for router

    def get_first_backbone_features(self, images):
        # Extract features from B1 backbone
        features_list = []

        with torch.no_grad():
            for img in images:
                # Add batch dimension: [1, 3, H, W]
                feats = self.b1.backbone(img.unsqueeze(0)) # Dict[str, Tensor]
                
                # Concatenate all FPN level features
                concat_feats = []
                for feat_map in feats.values():
                    # Apply adaptive pooling to each feature map
                    pooled = self.adaptive_pool(feat_map) # [1, C, 1, 1]
                    concat_feats.append(pooled.squeeze()) # [C]
                
                # Concatenate features from all FPN levels
                combined = torch.cat(concat_feats, dim=0) # [total_C]
                features_list.append(combined)

        # Stack into [B, total_C]
        features = torch.stack(features_list, dim=0)
        return features
    
    def forward(self, images, targets, train_router):
        '''
        Params:
            images: list[Tensor]
            targets: list[dict] or None
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
                    
                per_loss_b1 = torch.stack(per_loss_b1)
                per_loss_b2 = torch.stack(per_loss_b2)

                # 1 means go to B2, 0 means exit
                margin = 0.1
                routing_label = (per_loss_b2 + margin < per_loss_b1).float()
                routing_label = routing_label.view(-1, 1)

                features = self.get_first_backbone_features(images)
                score = self.router(features)
                # print("Router output shape:", score.shape)

                # BCE loss
                bce = F.binary_cross_entropy(score, routing_label)

                outputs["Loss_Router"] = bce
                outputs["Router"] = routing_label
                outputs["Router_Score"] = score.detach()
            
            # print(f"Outputs: {outputs}")
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

def train_backbones(model, train_loader, epochs):
    # Train both backbones
    params = [p for p in model.b1.parameters() if p.requires_grad] + \
             [p for p in model.b2.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = sanitize_boxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images, targets, train_router=False)

            loss = outputs['Loss_B1'] + outputs['Loss_B2']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")

    print("Finished training backbones")

    return model

def train_router(model, train_loader, epochs):
    # Train router
    params = [p for p in model.router.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = sanitize_boxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images, targets, train_router=True)

            loss = outputs['Loss_Router']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
            # Calculate routing accuracy
            routing_pred = (outputs['Router_Score'] >= 0.5).float()
            routing_label = outputs['Router']
            correct = (routing_pred == routing_label).sum().item()
            total_correct += correct
            total_samples += len(routing_label)
        
        avg_accuracy = total_correct / total_samples
        print(f"Epoch [{epoch + 1}/{epochs}] Avg Loss: {total_loss / len(train_loader):.4f} "
              f"Avg Accuracy: {avg_accuracy:.2%}")

    return model

def compute_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def visualize_detections(image_tensor, detections, save_path, confidence_threshold, 
                        backbone_used, show_gt, gt_boxes, gt_labels):
    # Convert tensor to numpy array
    img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    
    # Denormalize if needed (assuming ImageNet normalization)
    # If your images are already in [0,1], skip this
    img_np = np.clip(img_np, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    
    boxes = detections['boxes'].cpu()
    labels = detections['labels'].cpu()
    scores = detections['scores'].cpu()
    
    # Draw predictions
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle((x1, y1), width, height, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            class_name = COCO_CLASSES.get(int(label), f'class_{label}')
            text = f'{class_name}: {score:.2f}'
            ax.text(x1, y1 - 5, text, color='red', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
    
    # Draw ground truth boxes if requested
    if show_gt and gt_boxes is not None:
        gt_boxes_np = gt_boxes.cpu() if isinstance(gt_boxes, torch.Tensor) else gt_boxes
        gt_labels_np = gt_labels.cpu() if isinstance(gt_labels, torch.Tensor) else gt_labels
        
        for box, label in zip(gt_boxes_np, gt_labels_np):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height, 
                                     linewidth=2, edgecolor='green', 
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            ax.text(x1, y2 + 15, f'GT: {class_name}', color='green', 
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))
    
    # Add title with backbone info
    title = f'Detections (Backbone: {backbone_used})'
    if show_gt:
        title += ' | Red=Predictions, Green=Ground Truth'
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test(model, test_loader, confidence_threshold):
    model.eval()
    correct = 0
    b1_count = 0
    b2_count = 0
    total = 0
    total_tp = 0
    total_fp = 0
    total_gt = 0
    images_saved = 0
    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = list(image.to(device) for image in images)

            # Get routing decisions
            features = model.get_first_backbone_features(images)
            scores = model.router(features)

            routing_decisions = []
            for score in scores:
                total += 1
                if score.item() < model.threshold:
                    b1_count += 1
                    routing_decisions.append("B1")
                else:
                    b2_count += 1
                    routing_decisions.append("B2")

            detections = model(images, targets=None, train_router=False)
            print("Detection output example:", detections[0])

            for img_idx, (image, detection, target, backbone) in enumerate(
                zip(images, detections, targets, routing_decisions)
            ):
                boxes = detection['boxes']
                labels = detection['labels']
                det_scores = detection['scores']
                
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                # Filter detections by confidence
                mask = det_scores > confidence_threshold
                boxes = boxes[mask]
                labels = labels[mask]
                det_scores = det_scores[mask]
                
                # Calculate metrics (simplified - matches predictions to GT by IoU)
                matched_gt = set()
                for pred_box, pred_label in zip(boxes, labels):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    # Find best matching GT box
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_label == pred_label: # Same class
                            iou = compute_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    
                    # Consider it a match if IoU > 0.5
                    if best_iou > 0.5 and best_gt_idx not in matched_gt:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1
                
                total_gt += len(gt_boxes)
                
                # Save visualizations
                if images_saved < 50:
                    save_path = os.path.join(
                        "results", 
                        f'batch{batch_idx}_img{img_idx}_{backbone}.png'
                    )
                    
                    # Reconstruct detection dict for visualization
                    vis_detection = {
                        'boxes': boxes,
                        'labels': labels,
                        'scores': det_scores
                    }
                    
                    visualize_detections(
                        image, 
                        vis_detection, 
                        save_path,
                        confidence_threshold=confidence_threshold,
                        backbone_used=backbone,
                        show_gt=True,
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels
                    )
                    images_saved += 1
                    print(f"Saved visualization: {save_path}")

    print(f"Routed to B1 (lightweight): {b1_count} ({b1_count/total:.1%})")
    print(f"Routed to B2 (heavy): {b2_count} ({b2_count/total:.1%})")

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    
    print(f"\nDetection Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0:.3f}")
    print(f"Accuracy: {correct / total}")

if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Currently using the 5k subset of COCO from Kaggle
    subset_size = 100 # Use smaller portion
    split_annotations_path = "coco/annotations/instances_train2017.json"

    # Split dataset into smaller train/val subsets if not already done
    train_subset_path = f"coco/annotations/instances_train2017_{subset_size}.json"
    val_subset_path = f"coco/annotations/instances_val2017_{subset_size}.json"

    if not (os.path.exists(train_subset_path) and os.path.exists(val_subset_path)):
        print(f"Creating reduced dataset of {subset_size} images...")
        split_dataset(split_annotations_path, size=subset_size)
    else:
        print(f"Subset already exists for {subset_size} images.")

    # train_path = "coco/annotations/instances_train2017.json"
    num_classes, num_images_train = process_dataset(train_subset_path)
    # get_annotation_info(train_path)

    # Hyperparameters
    # batch_size = 10
    # num_workers = 2

    # # Train Dataset
    # train_root = f"coco/train2017_{subset_size}"
    # train_annotations = train_subset_path
    # train_dataset = CocoDataset(
    #     root=train_root,
    #     annotation=train_annotations,
    #     transform=get_transform()
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     collate_fn=collate_fn
    # )

    # # Test dataset
    # val_root = f"coco/val2017_{subset_size}"
    # val_annotations = val_subset_path
    # val_dataset = CocoDataset(
    #     root=val_root,
    #     annotation=val_annotations,
    #     transform=get_transform()
    # )
    # test_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=collate_fn
    # )

    # # Instantiate model
    # model = DynamicDet(num_classes=num_classes).to(device)

    # backbone_epochs = 2
    # router_epochs = 2
    # confidence_threshold = 0.1

    # # Train and Test

    # # train_backbone_start = time.time()
    # # model = train_backbones(model, train_loader, backbone_epochs)
    # # train_backbone_end = time.time()
    # # train_backbone_time = train_backbone_end - train_backbone_start
    # # print(f"Backbone training time: {train_backbone_time:.2f} seconds")

    # model.b1.eval()
    # model.b2.eval()

    # train_router_start = time.time()
    # model = train_router(model, train_loader, router_epochs)
    # train_router_end = time.time()
    # train_router_minutes = (train_router_end - train_router_start) / 60
    # train_router_seconds = (train_router_end - train_router_start) % 60
    # print(f"Router training time: {train_router_minutes:.0f} minutes {train_router_seconds:.2f} seconds")

    # test(model, test_loader, confidence_threshold)

    # end_time = time.time()
    # minutes = (end_time - start_time) / 60
    # seconds = (end_time - start_time) % 60
    # print(f"Total time: {minutes:.0f} minutes {seconds:.2f} seconds")
