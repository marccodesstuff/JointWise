import torch
import torch.nn as nn
from torchvision import models

import timm

class BaseModelBuilder:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def build_resnext50(self, dropout_rate=0.5, freeze_backbone=True):
        try:
            model = models.resnext50_32x4d(pretrained=True)
        except Exception:
            model = models.resnext50_32x4d(pretrained=False)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        return model

    def build_densenet201(self, dropout_rate=0.5, freeze_backbone=True):
        try:
            model = models.densenet201(pretrained=True)
        except Exception:
            model = models.densenet201(pretrained=False)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        return model

    def build_efficientnet_b0(self, dropout_rate=0.5, freeze_backbone=True):
        try:
            model = models.efficientnet_b0(pretrained=True)
        except Exception:
            model = models.efficientnet_b0(pretrained=False)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        return model

    def build_xception(self, dropout_rate=0.5, freeze_backbone=True):
        model = timm.create_model('xception41', pretrained=True)
        # Prefer timm reset_classifier API if available
        if hasattr(model, 'get_classifier') and hasattr(model, 'reset_classifier'):
            in_ch = model.get_classifier().in_features
            model.reset_classifier(num_classes=self.num_classes, global_pool='avg')
            if freeze_backbone:
                # Freeze all but classifier params
                clf_names = set()
                # Try to infer names used by timm for classifier params
                for name, _ in model.named_parameters():
                    if 'classifier' in name or 'head' in name or name.endswith('fc.weight') or name.endswith('fc.bias'):
                        parts = name.split('.')
                        if parts:
                            clf_names.add(parts[0])
                for name, param in model.named_parameters():
                    if not ('classifier' in name or 'head' in name or name.endswith('fc.weight') or name.endswith(
                            'fc.bias')):
                        param.requires_grad = False
            return model
        # Fallback manual head replacement
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
            num_features = model.classifier.in_features
            new_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, self.num_classes)
            )
            model.classifier = new_head
        elif hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            num_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, self.num_classes)
            )
        elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        else:
            raise AttributeError("Unsupported xception head layout")
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not any(k in name for k in ['classifier', 'head', 'fc']):
                    param.requires_grad = False
        return model

    def get_model_by_name(self, model_name, **kwargs):
        if model_name == 'resnext50':
            return self.build_resnext50(**kwargs)
        elif model_name == 'densenet201':
            return self.build_densenet201(**kwargs)
        elif model_name == 'efficientnet_b0':
            return self.build_efficientnet_b0(**kwargs)
        elif model_name == 'xception':
            return self.build_xception(**kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")