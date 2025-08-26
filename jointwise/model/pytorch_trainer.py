import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

class PyTorchTrainer:
    def __init__(self, model, device, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            train_bar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs=50, learning_rate=1e-4,
              patience=10, save_path=None):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Robust scheduler across torch versions
        try:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        except TypeError:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        best_val_acc = 0.0
        patience_counter = 0
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc = self.validate_epoch(val_loader)
            scheduler.step(val_acc)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
        return self.training_history