# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import TissueMNIST


"""
Let's try a completely different approach focusing on the model's learning dynamics:
Focal Loss instead of CrossEntropyLoss to handle class imbalance better
"""


class TissueCNN(nn.Module):
    def __init__(self):
        super(TissueCNN, self).__init__()
        
        # Consistent dropout rate as mentioned in paper
        self.dropout = nn.Dropout(0.2)  # Lower dropout for 2D
        
        self.features = nn.Sequential(
            # First block - maintain spatial information early
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),  # Remove bias before BatchNorm
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            self.dropout,
            nn.MaxPool2d(2),  # Delayed pooling
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            self.dropout,
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            self.dropout,
            nn.MaxPool2d(2),
        )
        
        # Classifier maintaining more connections
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512, bias=False),  # Adjusted for delayed pooling
            nn.BatchNorm1d(512),  # Add BN to dense layers
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            self.dropout,
            nn.Linear(512, 8)  # Direct mapping to output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, num_epochs=10, use_wandb=False, patience=1):
    if use_wandb:
        wandb.init(project="tissue-mnist", name="tissue_classifier")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Simpler training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher learning rate
    
    # Replace the existing scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Add label smoothing to criterion
    criterion = FocalLoss(gamma=2)
    
    best_val_loss = float('inf')
    patience = patience
    patience_counter = 0
    best_model_state = None  # Add this line to store the best model state
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.squeeze().long().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
        
        # Update learning rate
        scheduler.step()
        
        # Check for early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            patience_counter = 0
            print(f'Validation loss decreased to {val_loss:.4f}. Saving model...')
        else:
            patience_counter += 1
            print(f'EarlyStopping counter: {patience_counter} out of {patience}')
            
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    if use_wandb:
        wandb.finish()
    
    # Load the best model state before returning
    model.load_state_dict(best_model_state)
    return model  # Return the model with best weights loaded

# Define transforms at module level
base_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

augment_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class AugmentedTissueMNIST(TissueMNIST):
    def __init__(self, split, transform=None, target_transform=None, download=True):
        super().__init__(split=split, transform=transform, target_transform=target_transform, download=download)
        
        # Calculate class distributions
        labels = [label.item() for label in self.labels]
        unique, counts = np.unique(labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))
        
        # Find the median class count instead of max
        self.target_count = int(np.median(list(self.class_counts.values())))
        
        # Calculate augmentation multipliers for each class
        # Only augment classes below the median
        self.multipliers = {
            cls: max(1, min(3, int(np.ceil(self.target_count / count))))
            for cls, count in self.class_counts.items()
            if count < self.target_count
        }
        
        # Create augmented dataset
        self.augmented_indices = []
        for idx in range(len(self.imgs)):
            label = self.labels[idx].item()
            # Only augment if class needs augmentation
            if label in self.multipliers:
                mult = self.multipliers[label]
                # Original image
                self.augmented_indices.append((idx, False))
                # Augmented copies (limited)
                for _ in range(mult - 1):
                    self.augmented_indices.append((idx, True))
            else:
                # Just add original image for well-represented classes
                self.augmented_indices.append((idx, False))
    
    def __len__(self):
        return len(self.augmented_indices)
    
    def __getitem__(self, idx):
        orig_idx, should_augment = self.augmented_indices[idx]
        img = self.imgs[orig_idx]
        label = self.labels[orig_idx]
        
        # Convert numpy array to PIL Image
        img = transforms.ToPILImage()(img)
        
        if should_augment:
            img = augment_transform(img)
        else:
            img = base_transform(img)
            
        return img, label

def prepare_data(batch_size=32):
    # Remove transform definitions from here since they're now at module level
    
    # Create datasets with augmentation
    train_dataset = AugmentedTissueMNIST(split="train", download=True)
    val_dataset = TissueMNIST(split="val", transform=base_transform, download=True)
    
    # Print original and augmented dataset sizes
    print(f"Original training set size: {len(train_dataset.imgs)}")
    print(f"Augmented training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Calculate and print class distribution after augmentation
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    val_labels = [label.item() for _, label in val_dataset]
    
    def print_class_distribution(labels, name):
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"\n{name} class distribution:")
        for class_idx, count in dist.items():
            print(f"Class {class_idx}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    print_class_distribution(train_labels, "Augmented training")
    print_class_distribution(val_labels, "Validation")
    
    # Create test dataset
    test_dataset = TissueMNIST(split="test", transform=base_transform, download=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = FocalLoss(gamma=2)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print('-' * 60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(batch_size=32)
    
    # Create and train model
    model = TissueCNN()
    trained_model = train_model(model, train_loader, val_loader, use_wandb=False)
    
    # Save the trained model's state dict
    torch.save(trained_model.state_dict(), 'tissue_classifier.pth')
    
    # Evaluate model (no need to load state dict again since trained_model already has the best weights)
    evaluate_model(trained_model, test_loader)