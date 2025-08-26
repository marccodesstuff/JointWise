from pathlib import Path
import pandas as pd

# Load and process annotations from knee.csv
class DataProcessor:
    def __init__(self, csv_path, png_dir):
        self.csv_path = Path(csv_path)
        self.png_dir = Path(png_dir)
        self.df = None
        self.subject_labels = {}

    def load_annotations(self):
        """Load and process knee annotations"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} annotations")
        print(f"Unique files: {self.df['file'].nunique()}")
        print(f"Label distribution:")
        print(self.df['label'].value_counts())
        return self.df

    def create_target_labels(self):
        """Create target labels: ACL tear, Meniscus tear, Neither - treating cases with both as separate entries"""
        # Map detailed labels to our target classes
        acl_keywords = ['Ligament - ACL High Grade Sprain', 'Ligament - ACL Low Grade sprain']
        meniscus_keywords = ['Meniscus Tear']

        subject_conditions = {}

        for file_id in self.df['file'].unique():
            file_data = self.df[self.df['file'] == file_id]
            labels = file_data['label'].tolist()

            has_acl = any(any(keyword.lower() in label.lower() for keyword in acl_keywords) for label in labels)
            has_meniscus = any(
                any(keyword.lower() in label.lower() for keyword in meniscus_keywords) for label in labels)

            # Instead of creating a "Both" class, we'll create separate entries
            if has_acl and has_meniscus:
                # Create two separate entries: one for ACL tear and one for Meniscus tear
                subject_conditions[f"{file_id}_ACL"] = 'ACL_tear'
                subject_conditions[f"{file_id}_Meniscus"] = 'Meniscus_tear'
            elif has_acl:
                subject_conditions[file_id] = 'ACL_tear'
            elif has_meniscus:
                subject_conditions[file_id] = 'Meniscus_tear'
            else:
                continue

        self.subject_labels = subject_conditions
        print("\nSubject-level label distribution:")
        label_counts = pd.Series(list(subject_conditions.values())).value_counts()
        print(label_counts)

        return subject_conditions

    def get_bounding_boxes(self, file_id, slice_num):
        """Get bounding boxes for a specific file and slice"""
        slice_data = self.df[(self.df['file'] == file_id) & (self.df['slice'] == slice_num)]
        boxes = []
        for _, row in slice_data.iterrows():
            boxes.append({
                'x': row['x'], 'y': row['y'],
                'width': row['width'], 'height': row['height'],
                'label': row['label']
            })
        return boxes

    def get_available_images(self):
        available_images = []
        # Only keep rows with the target labels
        target_labels = ['Ligament - ACL High Grade Sprain', 'Ligament - ACL Low Grade sprain', 'Meniscus Tear']
        filtered_df = self.df[self.df['label'].isin(target_labels)]
        # Build a set of (file_id, slice, label) for fast lookup
        detection_set = set((row['file'], row['slice'], row['label']) for _, row in filtered_df.iterrows())
        for png_file in self.png_dir.glob('*.png'):
            filename = png_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                file_id = '_'.join(parts[:-1])
                slice_num = int(parts[-1])
                # Check for each target label
                for label in target_labels:
                    if (file_id, slice_num, label) in detection_set:
                        # Map to unified label
                        mapped_label = 'ACL_tear' if 'ACL' in label else 'Meniscus_tear'
                        available_images.append({
                            'path': str(png_file),
                            'file_id': file_id,
                            'slice': slice_num,
                            'label': mapped_label
                        })
        print(f"\nFound {len(available_images)} available images for ACL/Meniscus Tear")
        return available_images