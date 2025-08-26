import cv2
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

class MRIDatasetWithBboxes(Dataset):
    def __init__(self, data, data_processor, transform = None, num_class = 2, use_bboxes = True):
        self.data = data
        self.data_processor = data_processor
        self.transform = transform
        self.num_class = num_class
        self.use_bboxes = use_bboxes

        self.fixed_classes = np.array(["ACL_tear", "Meniscus_tear"])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.fixed_classes)
        all_labels = [item["label"] for item in self.data]
        self.encoded_labels = self.label_encoder.transform(self.label_encoder.fit_transform(all_labels))

    def extract_original_file_id(self, file_id):
        if file_id.endswith("_ACL") or file_id.endswith("_Meniscus"):
            return file_id.rsplit("_", 1)[0]
        return file_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not load image {item["path"]}")
            image = np.zeros((320, 320), dtype = np.uint8)

        image_rgb = np.stack([image, image, image], axis = -1)

        if self.use_bboxes:
            original_file_id = self.extract_original_file_id(item["file_id"])
            boxes = self.data_processor.get_bboxes(original_file_id, item["slice"])

            bboxes = []
            bbox_labels = []

            for box in boxes:
                x_min = box["x"]
                y_min = box["y"]
                x_max = box["x"] + box["width"]
                y_max = box["y"] + box["height"]
                h, w = image_rgb.shape[:2]
                x_min = max(0, min(x_min, w - 1))
                y_min = max(0, min(y_min, h - 1))
                x_max = min(x_max, w - 1)
                y_max = min(y_max, h - 1)

                bboxes.append([x_min, y_min, x_max, y_max])
                bbox_labels.append(box["label"])

            if self.transform and len(bboxes) > 0:
                try:
                    augmented = self.transform(
                        image = image_rgb,
                        bboxes = bboxes,
                        labels = bbox_labels
                    )
                except Exception as e:
                    print(f"Augmentation failed for image {idx}: {e}")

                    if self.transform:
                        augmented = self.transform(
                            image = image_rgb,
                            bboxes = [],
                            bbox_labels = []
                        )
                        image_rgb = augmented["image"]

                    transformed_bboxes = bboxes
                    transformed_labels = bbox_labels
            else:
                if self.transform:
                    augmented = self.transform(
                        image = image_rgb,
                        bboxes = [],
                        bbox_labels = []
                    )
                    image_rgb = augmented["image"]

                transformed_bboxes = bboxes
                transformed_labels = bbox_labels

            # Normalize bounding boxes to [0, 1] after the image augmentation
            h, w = image_rgb.shape[1], image_rgb.shape[2] if isinstance(image_rgb, torch.Tensor) else image_rgb.shape[:2]

            normalized_bboxes = []

            for bboxes in transformed_bboxes:
                x_min, y_min, x_max, y_max = bboxes
                normalized_bboxes.append([
                    x_min / w,
                    y_min / h,
                    x_max / w,
                    y_max / h
                ])
        else:
            if self.transform:
                if hasattr(self.transform, "bbox_params"):
                    augmented = self.transform(
                        image = image_rgb,
                        bboxes = [],
                        bbox_labels = []
                    )
                else:
                    augmented = self.transform(
                        image = image_rgb,
                    )

            # transformed_bboxes = []
            transformed_labels = []
            normalized_bboxes = []

        if isinstance(image_rgb, np.ndarray):
            image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
        else:
            image_tensor = image_rgb.float() if hasattr(image_rgb, "float") else image_rgb

        label = self.encoded_labels[idx]

        if self.use_bboxes:
            return {
                "image": image_tensor,
                "label": label,
                "bboxes": normalized_bboxes,
                "bbox_labels": transformed_labels,
                "original_file_id": self.extract_original_file_id(item["file_id"]),
                "slice": item["slice"]
            }
        else:
            return image_tensor, label

    def get_labels(self):
        return [item["label"] for item in self.data]

    def get_class_weights(self):
        labels = []

        for i in range(len(self)):
            _, label = self[i]
            labels.append(label)

        class_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)
        weights = []

        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)
            weight = total_samples / (num_classes * count)
            weights.append(weight)

        return torch.tensor(weights, dtype = torch.float32)