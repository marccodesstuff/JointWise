import albumentations as A
from albumentations.pytorch import ToTensorV2

class MRIAugmentationWithBboxes:
    def __init__(self, image_size = (320, 320)):
        self.image_size = image_size

    def get_train_augmentation_with_bboxes(self):
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1], p = 1),
            A.HorizontalFlip(p = 0.5),
            A.Rotate(limit = 10, p = 0.3),
            A.ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.05, rotate_limit = 5, p = 0.3),
            ToTensorV2()
        ], bbox_params = A.BboxParams(
            format = "pascal_voc",
            min_area = 0,
            min_visibility = 0.1,
            label_fields = ["bbox_labels"]
        ))

    def get_val_augmentation_with_bboxes(self):
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1], p = 1),
            ToTensorV2()
        ], bbox_params = A.BboxParams(
            format = "pascal_voc",
            min_area = 0,
            min_visibility = 0.1,
            label_fields = ["bbox_labels"]
        ))