import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

class MultiTaskTrainer:
    """
    Trainer for multi-task learning with classification and bounding box regression
    """

    def __init__(self, model, device, class_weights=None,
                 classification_weight=1.0, bbox_weight=1.0):
        self.model = model.to(device)
        self.device = device
        self.classification_weight = classification_weight
        self.bbox_weight = bbox_weight

        # Loss functions
        if class_weights is not None:
            self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.classification_criterion = nn.CrossEntropyLoss()

        # For bbox regression, we'll use SmoothL1Loss (Huber loss)
        # which is less sensitive to outliers than MSE
        self.bbox_criterion = nn.SmoothL1Loss()

        # Training history
        self.training_history = {
            'train_loss': [], 'train_class_loss': [], 'train_bbox_loss': [],
            'train_acc': [], 'val_loss': [], 'val_class_loss': [],
            'val_bbox_loss': [], 'val_acc': []
        }

    def train_epoch(self, train_loader, optimizer):
        self.model.train()

        running_total_loss = 0.0
        running_class_loss = 0.0
        running_bbox_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc='Training')

        for batch_idx, batch_data in enumerate(train_bar):
            # Extract data from batch
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            bboxes = batch_data['bboxes'].to(self.device)  # [batch_size, 4]

            optimizer.zero_grad()

            # Forward pass
            class_output, bbox_output = self.model(images)

            # Calculate losses
            class_loss = self.classification_criterion(class_output, labels)
            bbox_loss = self.bbox_criterion(bbox_output, bboxes)

            # Combined loss
            total_loss = (self.classification_weight * class_loss +
                          self.bbox_weight * bbox_loss)

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update metrics
            running_total_loss += total_loss.item()
            running_class_loss += class_loss.item()
            running_bbox_loss += bbox_loss.item()

            _, predicted = torch.max(class_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_bar.set_postfix({
                'Total Loss': f'{running_total_loss / (batch_idx + 1):.4f}',
                'Class Loss': f'{running_class_loss / (batch_idx + 1):.4f}',
                'BBox Loss': f'{running_bbox_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        epoch_total_loss = running_total_loss / len(train_loader)
        epoch_class_loss = running_class_loss / len(train_loader)
        epoch_bbox_loss = running_bbox_loss / len(train_loader)
        epoch_acc = correct / max(total, 1)

        return epoch_total_loss, epoch_class_loss, epoch_bbox_loss, epoch_acc

    def validate_epoch(self, val_loader):
        self.model.eval()

        running_total_loss = 0.0
        running_class_loss = 0.0
        running_bbox_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                bboxes = batch_data['bboxes'].to(self.device)

                # Forward pass
                class_output, bbox_output = self.model(images)

                # Calculate losses
                class_loss = self.classification_criterion(class_output, labels)
                bbox_loss = self.bbox_criterion(bbox_output, bboxes)
                total_loss = (self.classification_weight * class_loss +
                              self.bbox_weight * bbox_loss)

                # Update metrics
                running_total_loss += total_loss.item()
                running_class_loss += class_loss.item()
                running_bbox_loss += bbox_loss.item()

                _, predicted = torch.max(class_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_total_loss = running_total_loss / len(val_loader)
        epoch_class_loss = running_class_loss / len(val_loader)
        epoch_bbox_loss = running_bbox_loss / len(val_loader)
        epoch_acc = correct / max(total, 1)

        return epoch_total_loss, epoch_class_loss, epoch_bbox_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs=50, learning_rate=1e-4,
              patience=10, save_path=None):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        try:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                          patience=5, verbose=True)
        except TypeError:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_val_acc = 0.0
        patience_counter = 0

        print(f"Starting multi-task training for {epochs} epochs...")
        print(f"Loss weights - Classification: {self.classification_weight}, BBox: {self.bbox_weight}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Training
            train_total_loss, train_class_loss, train_bbox_loss, train_acc = self.train_epoch(train_loader, optimizer)

            # Validation
            val_total_loss, val_class_loss, val_bbox_loss, val_acc = self.validate_epoch(val_loader)

            # Update scheduler
            scheduler.step(val_acc)

            # Store history
            self.training_history['train_loss'].append(train_total_loss)
            self.training_history['train_class_loss'].append(train_class_loss)
            self.training_history['train_bbox_loss'].append(train_bbox_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_total_loss)
            self.training_history['val_class_loss'].append(val_class_loss)
            self.training_history['val_bbox_loss'].append(val_bbox_loss)
            self.training_history['val_acc'].append(val_acc)

            # Print epoch results
            print(f"Train - Total: {train_total_loss:.4f}, Class: {train_class_loss:.4f}, "
                  f"BBox: {train_bbox_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Total: {val_total_loss:.4f}, Class: {val_class_loss:.4f}, "
                  f"BBox: {val_bbox_loss:.4f}, Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
        return self.training_history