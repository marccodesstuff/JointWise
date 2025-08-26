import pandas as pd
from sklearn.model_selection import train_test_split

class ImprovedSubjectLevelSplitter:
    def __init__(self, available_images, test_size = 0.2, val_size = 0.2, random_state = 42):
        self.available_images = available_images
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def extract_original_file_id(self, file_id):
        if file_id.endswith('_ACL') or file_id.endswith('_Meniscus'):
            return file_id.rsplit('_', 1)[0]
        return file_id

    def split_subjects(self):
        subject_data = {}
        original_to_entries = {}

        for img_info in self.available_images:
            file_id = img_info['file_id']
            original_file_id = self.extract_original_file_id(file_id)

            if original_file_id not in subject_data:
                subject_data[original_file_id] = []
                original_to_entries[original_file_id] = []

            subject_data[original_file_id].append(img_info)
            if file_id not in original_to_entries[original_file_id]:
                original_to_entries[original_file_id].append(file_id)

        # For stratification, handle subjects that have both ACL and Meniscus
        original_subjects = list(subject_data.keys())
        subject_stratify_labels = []

        for original_file_id in original_subjects:
            # Check what types of entries this subject has
            entries = original_to_entries[original_file_id]
            has_acl = any('_ACL' in entry for entry in entries)
            has_meniscus = any('_Meniscus' in entry for entry in entries)
            has_direct = any(entry == original_file_id for entry in entries)

            # Determine stratification label
            if has_acl:
                subject_stratify_labels.append('ACL_tear')
            elif has_meniscus:
                subject_stratify_labels.append('Meniscus_tear')
            elif has_direct:
                # Use the label from the direct entry
                direct_entries = [img for img in subject_data[original_file_id] if img['file_id'] == original_file_id]
                if direct_entries:
                    subject_stratify_labels.append(direct_entries[0]['label'])
                else:
                    subject_stratify_labels.append('Neither')
            else:
                subject_stratify_labels.append('Neither')

        print(f"Total original subjects: {len(original_subjects)}")
        print(f"Subject stratification distribution:")
        print(pd.Series(subject_stratify_labels).value_counts())

        # First split: train+val vs test (based on original subjects)
        train_val_subjects, test_subjects = train_test_split(
            original_subjects,
            test_size=self.test_size,
            stratify=subject_stratify_labels,
            random_state=self.random_state
        )

        # Second split: train vs val
        train_val_labels = [subject_stratify_labels[original_subjects.index(subj)] for subj in train_val_subjects]
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=self.val_size / (1 - self.test_size),  # Adjust for already removed test set
            stratify=train_val_labels,
            random_state=self.random_state
        )

        # Create final datasets (including all entries for each original subject)
        train_data = []
        val_data = []
        test_data = []

        for subj in train_subjects:
            train_data.extend(subject_data[subj])
        for subj in val_subjects:
            val_data.extend(subject_data[subj])
        for subj in test_subjects:
            test_data.extend(subject_data[subj])

        print(f"\nData split completed:")
        print(f"Train: {len(train_data)} images from {len(train_subjects)} original subjects")
        print(f"Val: {len(val_data)} images from {len(val_subjects)} original subjects")
        print(f"Test: {len(test_data)} images from {len(test_subjects)} original subjects")

        # Print label distribution for each split
        for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            labels = [item['label'] for item in split_data]
            print(f"\n{split_name} label distribution:")
            print(pd.Series(labels).value_counts())

        return train_data, val_data, test_data