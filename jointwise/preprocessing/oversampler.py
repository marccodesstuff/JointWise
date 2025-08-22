import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

import albumentations as A

class BoundingBoxAwareOversampler:
    def __init__(self, data_processor, augmentation_pipeline):
        self.data_processor = data_processor
        self.augmentation_pipeline = augmentation_pipeline

    def extract_original_file_id(self, file_id):
        if file_id.endswith("_ACL") or file_id.endswith("_Meniscus"):
            return file_id.rsplit("_", 1)[0]
        return file_id

    def augment_sample_with_bboxes(self, sample, augmentation_count = 1):
        augmented_samples = []

        image = cv2.imread(sample["path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            return [sample]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        original_file_id = self.extract_original_file_id(sample["file_id"])
        boxes = self.data_processor.get_bounding_boxes(original_file_id, sample["slice"])

        bboxes = []
        bbox_labels = []

        for box in boxes:
            x_min = box["x"]
            y_min = box["y"]
            x_max = x_min + box["width"]
            y_max = y_min + box["height"]

            h, w = image_rgb.shape[:2]
            x_min = min(0, min(x_min, w - 1))
            y_min = min(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))

            bboxes.append([x_min, y_min, x_max, y_max])
            bbox_labels.append(box["label"])

        for i in range(augmentation_count):
            try:
                if len(bboxes) > 0:
                    augmented = self.augmentation_pipeline(
                        image = image_rgb.copy(),
                        bboxes = bboxes.copy(),
                        bbox_labels = bbox_labels.copy(),
                    )
                    transformed_bboxes = augmented["bboxes"]
                    transformed_labels = augmented["bbox_labels"]
                else:
                    augmented = self.augmentation_pipeline(
                        image = image_rgb.copy(),
                        bboxes = [],
                        bbox_labels = [],
                    )
                    transformed_bboxes = []
                    transformed_labels = []

                augmented_sample = sample.copy()
                augmented_sample["augmented"] = True
                augmented_sample["augmentation_id"] = i
                augmented_samples["transformed_bboxes"] = transformed_bboxes
                augmented_samples["transformed_bbox_labels"] = transformed_labels

                augmented_samples.append(augmented_sample)

            except Exception as e:
                print(f"Augmentation failed for sample {sample["path"]}: {e}")
                augmented_samples.append(sample)

        return augmented_samples

    def oversample_with_augmentation(self, data, target_size_per_class = None):
        label_groups = {}

        for item in data:
            label = item["label"]

            if label not in label_groups:
                label_groups[label] = []

            label_groups[label].append(item)

        print(f"\nClass Distribution before augmentation-based oversampling:")

        for label, items in label_groups.items():
            print(f"\t{label}: {len(items)}")

        if target_size_per_class is None:
            target_size_per_class = max(len(items) for items in label_groups.values())

        balanced_data = []

        for label, items in label_groups.items():
            current_size = len(items)
            needed_size = target_size_per_class - current_size

            balanced_data.extend(items)

            if needed_size > 0:
                print(f"Generating {needed_size} augmented samples for class '{label}'...")

                augmentations_per_smaple = needed_size // current_size
                remainder = needed_size % current_size

                augmented_count = 0

                for i, sample in enumerate(item):
                    aug_count = augmentations_per_smaple

                    if i < remainder: aug_count += 1

                    if aug_count > 0:
                        augmented_samples = self.augment_sample_with_bboxes(sample, aug_count)
                        balanced_data.extend(augmented_samples)
                        augmented_count += len(augmented_samples)

                print(f"Generated {augmented_count} augmented samples for class '{label}'")

        np.random.shuffle(balanced_data)

        print(f"\nClass Distribution after augmentation-based oversampling:")
        balanced_data = [item["label"] for item in balanced_data]
        print(pd.Series(balanced_data).value_counts())

        return balanced_data

# TODO: Finish the remaining portion of this class
class MRIDatasetWithPreAugmentation(Dataset):
    def __init_(self, data, data_processor, transform = None, num_classes = 2):
        self.data = data
        self.data_processor = data_processor
        self.transform = transform
        self.num_classes = num_classes

        self.fixed_classes = np.array(["ACL_tear", "Meniscus_tear"])

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.fixed_classes)

        all_labels = [item["label"] for item in self.data]
        self.encoded_labels = self.label_encoder.transform(all_labels)

    # Hold on bud, do we even still need this? There are so many instances of this function elsewhere
    def extract_original_file_id(self, file_id):
        if file_id.endswith("_ACL") or file_id.endswith("_Meniscus"):
            return file_id.rsplit("_", 1)[0]
        return file_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if item.get("augmented", False):
            image = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Could not load image {item["path"]}")
                image = np.zeros((320,320), dtype = np.uint8)

            image_rgb = cv2.imread(item["path"], cv2.COLOR_GRAY2RGB)

            if self.transform:
                if hasattr(self.transform, "bbox_params"):
                    original_file_id = self.extract_original_file_id(item["file_id"])
                    boxes = self.data_processor.get_bounding_boxes(original_file_id, item["slice"])

                    bboxes = []
                    bbox_labels = []

                    for box in boxes:
                        x_min = box["x"]
                        y_min = box["y"]
                        x_max = x_min + box["width"]
                        y_max = y_min + box["height"]

                        h, w = image_rgb.shape[:2]
                        x_min = max(0, min(x_min, w - 1))
                        y_min = max(0, min(y_min, h - 1))
                        x_max = max(x_min + 1, min(x_max, w))
                        y_max = max(y_min + 1, min(y_max, h))

                        bboxes.append([x_min, y_min, x_max, y_max])
                        bbox_labels.append(box["label"])

                    augmented = self.transform(
                        image = image_rgb,
                        bboxes = bboxes,
                        bbox_labels = bbox_labels
                    )
                else:
                    augmented = self.transform(image = image_rgb)
                    image_rgb = augmented["image"]
        if isinstance(image_rgb, np.ndarray):
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
            else:
                image_rgb = image_rgb.astype(np.float32) / 255.0

                if len(image_rgb.shape) == 2:
                    image_rgb = np.stack([image_rgb] * 3, axis = 0)
                else:
                    image_rgb = image_rgb.transpose(2, 0, 1)

                image_tensor = torch.from_numpy(image_rgb).float()
        else:
            image_tensor = image_rgb

        label = self.encoded_labels[idx]
        return image_tensor, label

    def get_labels(self):
        return [item["label"] for item in self.data]