import warnings
warnings.filterwarnings('ignore')

# PyTorch deep learning framework
import torch
import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    """
    Multi-task learning model that performs both:
    1. Classification (ACL tear vs Meniscus tear)
    2. Bounding box regression (localization of pathology)
    """

    def __init__(self, backbone_name='resnext50', num_classes=2, num_bbox_coords=4,
                 dropout_rate=0.5, freeze_backbone=True):
        super(MultiTaskModel, self).__init__()
        self.num_classes = num_classes
        self.num_bbox_coords = num_bbox_coords

        # Build backbone model
        if backbone_name == 'resnext50':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            # Remove the original classifier
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone_name == 'densenet201':
            self.backbone = models.densenet201(pretrained=True)
            self.feature_dim = self.backbone.classifier.in_features
            # Remove the original classifier
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.backbone.classifier[1].in_features
            # Remove the original classifier
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # In MultiTaskModel.__init__ (add after efficientnet_b0 block)
        elif backbone_name == 'xception':
            import timm
            self.backbone = timm.create_model('xception41', pretrained=True, features_only=True)
            self.feature_dim = self.backbone.feature_info[-1]['num_chs']
            # Add a global pooling layer if needed
            # self.backbone = nn.Sequential(self.backbone, nn.AdaptiveAvgPool2d(1)) # IDK Copilot just told me to remove this one
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_bbox_coords),
            nn.Sigmoid()  # Normalize bbox coordinates to [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        # If features is a list (Xception), take the last one
        if isinstance(features, list):
            features = features[-1]
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        shared_features = self.shared_layers(features)
        classification_output = self.classifier(shared_features)
        bbox_output = self.bbox_regressor(shared_features)
        return classification_output, bbox_output

    def get_features(self, x):
        """Extract shared features for analysis"""
        with torch.no_grad():
            features = self.backbone(x)
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
            shared_features = self.shared_layers(features)
        return shared_features