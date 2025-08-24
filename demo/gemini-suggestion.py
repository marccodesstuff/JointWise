# --- Essential Imports ---
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PyTorch deep learning framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# You will need to install the 'timm' library for the Xception model.
# In a new cell, run: !pip install timm
try:
    import timm # For Xception and other models
except ImportError:
    print("Warning: 'timm' library not found. Please install it using `!pip install timm` to use the Xception model.")

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Machine learning utilities
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Environment setup complete.")
print(f"Using device: {device}")


# --- Step 1: Data Processing Class (from your original notebook) ---
# This class correctly handles your annotations and file paths.

class DataProcessor:
    def __init__(self, csv_path, png_dir):
        self.csv_path = Path(csv_path)
        self.png_dir = Path(png_dir)
        self.df = None
        self.subject_labels = {}

    def load_annotations(self):
        """Load and process knee annotations"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} annotations from {self.csv_path}")
        return self.df

    def create_subject_table(self) -> pd.DataFrame:
        """Create a per-subject table with ACL/Meniscus presence and a stratification category.

        Returns a DataFrame with columns: subject_id, has_acl, has_meniscus, strat_cat, n_slices_annot.
        """
        acl_keywords = ['Ligament - ACL High Grade Sprain', 'Ligament - ACL Low Grade sprain', 'ACL']
        meniscus_keywords = ['Meniscus', 'Meniscus Tear']

        recs = []
        for file_id in self.df['file'].dropna().astype(str).unique():
            fdf = self.df[self.df['file'].astype(str) == file_id]
            labels = fdf['label'].fillna('').astype(str).tolist()
            has_acl = any(any(k.lower() in lab.lower() for k in acl_keywords) for lab in labels)
            has_meniscus = any(any(k.lower() in lab.lower() for k in meniscus_keywords) for lab in labels)
            if has_acl and has_meniscus:
                cat = 'both'
            elif has_acl:
                cat = 'acl_only'
            elif has_meniscus:
                cat = 'meniscus_only'
            else:
                cat = 'neither'
            recs.append({
                'subject_id': file_id,
                'has_acl': has_acl,
                'has_meniscus': has_meniscus,
                'strat_cat': cat,
                'n_slices_annot': fdf['slice'].nunique() if 'slice' in fdf.columns else None,
            })
        subj_df = pd.DataFrame.from_records(recs)
        # Filter out subjects with neither label, since we don't have boxes for them downstream
        keep = subj_df['has_acl'] | subj_df['has_meniscus']
        subj_df = subj_df[keep].reset_index(drop=True)
        print("\nSubject-level label distribution:")
        print(subj_df['strat_cat'].value_counts())
        return subj_df

    def get_bounding_boxes(self, file_id, slice_num):
        """Get bounding boxes for a specific file and slice"""
        slice_data = self.df[(self.df['file'] == file_id) & (self.df['slice'] == slice_num)]
        boxes = []
        for _, row in slice_data.iterrows():
            # IMPORTANT: We map your string labels to integer IDs for the model.
            label = row['label']
            if 'ACL' in label:
                label_id = 1 # 1 for ACL
            elif 'Meniscus' in label:
                label_id = 2 # 2 for Meniscus
            else:
                continue # Skip other labels for now
            
            boxes.append({
                'x': row['x'], 'y': row['y'],
                'width': row['width'], 'height': row['height'],
                'label_id': label_id
            })
        return boxes
    
    def get_available_images(self, subjects: set[str] | None = None):
        """List slice images with at least one ACL or Meniscus box.

        Returns a list of dicts: {subject_id, slice, path, has_acl, has_meniscus}.
        If `subjects` is provided, only include those subject_ids.
        """
        all_data = []
        missing_images = []

        subject_ids = self.df['file'].dropna().astype(str).unique().tolist()
        for original_file_id in subject_ids:
            if subjects is not None and original_file_id not in subjects:
                continue

            pattern = f"{original_file_id}_*.png"
            found_images = False
            for image_path in self.png_dir.glob(pattern):
                filename = image_path.stem
                slice_str = filename.split('_')[-1]
                try:
                    slice_num = int(slice_str)
                except ValueError:
                    continue

                boxes = self.get_bounding_boxes(original_file_id, slice_num)
                if not boxes:
                    continue
                has_acl = any(b['label_id'] == 1 for b in boxes)
                has_meniscus = any(b['label_id'] == 2 for b in boxes)
                all_data.append({
                    'subject_id': original_file_id,
                    'slice': slice_num,
                    'path': str(image_path),
                    'has_acl': has_acl,
                    'has_meniscus': has_meniscus,
                })
                found_images = True
            if not found_images:
                missing_images.append(original_file_id)

        print(f"Found {len(all_data)} images with bounding boxes")
        if missing_images:
            print(f"Warning: {len(missing_images)} subjects have no images with bounding boxes (first 5): {missing_images[:5]}")
        return all_data


# --- Step 2: Modified Dataset Class for Object Detection ---
# This class is fundamentally changed to return the data in a format
# that object detection models expect: a tensor of boxes and labels.

class MRIDatasetWithBBoxes(Dataset):
    def __init__(self, data, data_processor, transform=None):
        self.data = data
        self.data_processor = data_processor
        self.transform = transform
        # We need a fixed label map for the two pathologies
        self.label_map = {'ACL_tear': 1, 'Meniscus_tear': 2}
        self.fixed_classes = np.array(['ACL_tear', 'Meniscus_tear'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['path']

        # Load image and resize to a consistent size if needed
        # The model's RPN will handle different sizes, but a fixed input
        # size simplifies things.
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h0, w0 = image.shape[:2]
        image = cv2.resize(image, (224, 224))
        # Convert to RGB and normalize to [0,1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_rgb = image_rgb.astype(np.float32) / 255.0

        # Extract the original file ID to get bounding boxes
        original_file_id = item.get('subject_id') or item.get('file_id')
        slice_num = item['slice']

        boxes_from_processor = self.data_processor.get_bounding_boxes(original_file_id, slice_num)

        bboxes = []
        labels = []
        for box in boxes_from_processor:
            x_min, y_min, w, h = box['x'], box['y'], box['width'], box['height']
            x_max, y_max = x_min + w, y_min + h
            # Scale to resized 224x224 if needed
            sx = 224.0 / float(w0) if w0 else 1.0
            sy = 224.0 / float(h0) if h0 else 1.0
            bboxes.append([x_min * sx, y_min * sy, x_max * sx, y_max * sy])  # pascal_voc format
            labels.append(box['label_id'])

        # Apply augmentations (with bbox transforms) if provided
        if self.transform is not None:
            transformed = self.transform(image=image_rgb, bboxes=bboxes, labels=labels)
            image_rgb = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        # Convert to tensors expected by FasterRCNN
        if not bboxes:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor, "image_id": torch.tensor([idx])}

        # Ensure image is a torch tensor [C,H,W]
        if isinstance(image_rgb, torch.Tensor):
            image_tensor = image_rgb
        else:
            image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1))

        return image_tensor, target

# We need this custom collate function for the DataLoader
# because each image has a different number of bounding boxes.
def custom_collate(data):
    return tuple(zip(*data))


# --- Step 3: Model Creation with Custom Backbones ---
# This function allows you to plug in any of your desired architectures.

def create_detection_model(backbone_name, num_classes):
    """
    Creates a FasterRCNN model with a specified backbone.
    num_classes: 1 (background) + number of pathologies.
    """
    if backbone_name == 'xception':
        try:
            # Using timm for Xception since it's not in torchvision
            backbone = timm.create_model('xception', pretrained=True, features_only=True)
            # timm models return features as a list, so we need to wrap it
            class BackboneWithFPN(nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone
                    
                def forward(self, x):
                    features = self.backbone(x)
                    # Return as a dictionary with single key for FasterRCNN
                    return {'0': features[-1]}  # Use the last feature map
                    
            backbone = BackboneWithFPN(backbone)
            backbone.out_channels = 2048
        except NameError:
            raise ValueError("Xception model requires the 'timm' library to be installed.")
    elif backbone_name == 'resnext':
        backbone = torchvision.models.resnext50_32x4d(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Wrap to return dictionary format
        class BackboneWithFPN(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                
            def forward(self, x):
                features = self.backbone(x)
                return {'0': features}
                
        backbone = BackboneWithFPN(backbone)
        backbone.out_channels = 2048
    elif backbone_name == 'densenet':
        backbone = torchvision.models.densenet121(pretrained=True).features
        # Wrap to return dictionary format
        class BackboneWithFPN(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                
            def forward(self, x):
                features = self.backbone(x)
                return {'0': features}
                
        backbone = BackboneWithFPN(backbone)
        backbone.out_channels = 1024
    elif backbone_name == 'efficientnet':
        backbone = torchvision.models.efficientnet_b0(pretrained=True).features
        # Wrap to return dictionary format
        class BackboneWithFPN(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                
            def forward(self, x):
                features = self.backbone(x)
                return {'0': features}
                
        backbone = BackboneWithFPN(backbone)
        backbone.out_channels = 1280
    else:
        raise ValueError("Unsupported backbone name")

    # Define a custom AnchorGenerator. You can fine-tune these parameters.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Instantiate the FasterRCNN model with our custom backbone
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    return model


# --- Helpers: balanced sampling over slice categories ---
def build_sample_weights(data_list: list[dict]):
    """Compute per-sample weights to balance ACL-only, Meniscus-only, and Both.

    Returns a list of floats aligned with data_list order.
    """
    def cat_of(item):
        a, m = item.get('has_acl', False), item.get('has_meniscus', False)
        if a and m:
            return 'both'
        elif a:
            return 'acl_only'
        elif m:
            return 'meniscus_only'
        else:
            return 'neither'

    cats = [cat_of(it) for it in data_list]
    # Count categories present in data
    uniq, counts = np.unique(cats, return_counts=True)
    count_map = {u: c for u, c in zip(uniq, counts)}
    # Ignore 'neither' if any; give minimal weight
    inv = {k: (1.0 / c if c > 0 else 0.0) for k, c in count_map.items()}
    # Normalize weights so average ~1
    weights = [inv.get(c, 0.0) for c in cats]
    mean_w = np.mean(weights) if weights else 1.0
    if mean_w > 0:
        weights = [w / mean_w for w in weights]
    return weights, count_map


# --- Step 4: The Training Loop ---
# This is a full, working training function for a single model.
# You will run this for each of your four models.

# ... existing code ...
def train_model(model, dataloader, optimizer, num_epochs, model_name, grad_accum_steps=1):
    print(f"\n--- Training {model_name} ---")
    model.train()
    for epoch in range(num_epochs):
        # We'll accumulate gradients over `grad_accum_steps` batches to emulate a larger batch size
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (images, targets) in enumerate(dataloader):
            # Move images and targets to the device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # The model returns a dictionary of losses during training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            running_loss += loss_value

            # Scale loss by accumulation steps before backward for stable gradients
            (losses / grad_accum_steps).backward()

            # Step and zero gradients every grad_accum_steps
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}, Loss: {running_loss / ( (i % grad_accum_steps) + 1):.4f}")
                running_loss = 0.0

        # If number of batches isn't divisible by grad_accum_steps, make sure to step once more
        if len(dataloader) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Training for {model_name} complete.")
# ... existing code ...


# --- Step 5: Implementing the Ensemble Inference ---
# This function combines predictions from multiple models and refines them.

def ensemble_inference(image, models, iou_threshold=0.5):
    all_boxes = []
    all_scores = []
    all_labels = []

    # Run inference for each model
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions = model([image.to(device)])
        
        # Collect predictions from each model
        all_boxes.append(predictions[0]['boxes'])
        all_scores.append(predictions[0]['scores'])
        all_labels.append(predictions[0]['labels'])

    # Concatenate predictions from all models
    all_boxes = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Use Non-Maximum Suppression (NMS) to refine the predictions
    # NMS merges overlapping boxes and keeps the highest-scoring ones
    keep = torchvision.ops.nms(all_boxes, all_scores, iou_threshold=iou_threshold)
    
    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]
    final_labels = all_labels[keep]

    return {
        'boxes': final_boxes,
        'scores': final_scores,
        'labels': final_labels
    }


# --- Step 6: Main Execution Block ---
# This block demonstrates how to use all the functions together.

if __name__ == '__main__':
    # --- Data Paths (Update these to your file locations) ---
    csv_path = 'annotations/knee.csv' # Path to your annotations CSV file
    png_dir = 'png-output' # Path to the directory containing image folders
    
    # --- Data Loading and Splitting ---
    processor = DataProcessor(csv_path, png_dir)
    processor.load_annotations()
    subj_df = processor.create_subject_table()
    if subj_df.empty:
        print("ERROR: No subjects with ACL or Meniscus labels found.")
        exit(1)

    # Subject-level split with stratification on category
    subject_ids = subj_df['subject_id'].tolist()
    strat = subj_df['strat_cat'].tolist()
    train_subjects, val_subjects = train_test_split(
        subject_ids,
        test_size=0.2,
        random_state=42,
        stratify=strat if len(set(strat)) > 1 else None,
    )

    # Collect slice records for selected subjects
    train_data = processor.get_available_images(subjects=set(train_subjects))
    val_data = processor.get_available_images(subjects=set(val_subjects))
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("ERROR: Training or validation set is empty!")
        print("This might be due to missing image files or annotation mismatches.")
        exit(1)
    
    # Train-time photometric augmentation (avoid geom transforms to keep boxes valid)
    train_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], check_each_transform=False))
    
    # Create dataset instances
    train_dataset = MRIDatasetWithBBoxes(train_data, processor, transform=train_transform)
    val_dataset = MRIDatasetWithBBoxes(val_data, processor)

    # Create DataLoader instances
    # Balanced sampling across ACL-only, Meniscus-only, Both
    weights, count_map = build_sample_weights(train_data)
    print("Train slice category counts:", count_map)
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(train_dataset), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=sampler, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

    # --- Model Initialization ---
    num_pathologies = 2 # ACL and Meniscus
    num_classes = num_pathologies + 1 # Background (0), ACL (1), Meniscus (2)

    model_xception = create_detection_model('xception', num_classes).to(device)
    model_resnext = create_detection_model('resnext', num_classes).to(device)
    model_densenet = create_detection_model('densenet', num_classes).to(device)
    model_efficientnet = create_detection_model('efficientnet', num_classes).to(device)

    # Define optimizers for each model
    optimizer_xception = optim.Adam(model_xception.parameters(), lr=0.001)
    optimizer_resnext = optim.Adam(model_resnext.parameters(), lr=0.001)
    optimizer_densenet = optim.Adam(model_densenet.parameters(), lr=0.001)
    optimizer_efficientnet = optim.Adam(model_efficientnet.parameters(), lr=0.001)
    
    # --- Training Phase ---
    # You would train each model individually here
    train_model(model_xception, train_dataloader, optimizer_xception, num_epochs=10, model_name="Xception")
    train_model(model_resnext, train_dataloader, optimizer_resnext, num_epochs=10, model_name="ResNeXt")
    train_model(model_densenet, train_dataloader, optimizer_densenet, num_epochs=10, model_name="DenseNet")
    train_model(model_efficientnet, train_dataloader, optimizer_efficientnet, num_epochs=10, model_name="EfficientNet")
    
    # --- Ensemble Inference Phase ---
    # After training and saving your models, you can perform inference on a new image.
    # For this example, we'll use a single image from the validation set.
    sample_image, _ = val_dataset[0]
    
    # Put all models in a dlist for the ensemble function
    # NOTE: You must load your traine weights before this step
    # Example: model_xception.load_state_dict(torch.load('xception_weights.pth'))
    
    ensemble_models = [model_xception, model_resnext, model_densenet, model_efficientnet]
    
    print("\n--- Running Ensemble Inference on a sample image ---")
    final_predictions = ensemble_inference(sample_image, ensemble_models, iou_threshold=0.3)
    
    print("Final Predictions:")
    for i, (box, score, label) in enumerate(zip(final_predictions['boxes'], final_predictions['scores'], final_predictions['labels'])):
        print(f"Prediction {i+1}:")
        print(f"  - Bounding Box: {box.tolist()}")
        print(f"  - Score: {score.item():.4f}")
        print(f"  - Label: {label.item()}")