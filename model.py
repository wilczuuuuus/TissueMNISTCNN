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
Tried to augement to the same propotion -> reversed val/train acc so augument more naturally:
Augmented training class distribution:
Class 0: 53075 samples (25.95%)
Class 1: 15628 samples (7.64%)
Class 2: 17598 samples (8.61%)
Class 3: 15406 samples (7.53%)
Class 4: 23578 samples (11.53%)
Class 5: 15410 samples (7.54%)
Class 6: 39203 samples (19.17%)
Class 7: 24608 samples (12.03%)

Validation class distribution:
Class 0: 7582 samples (32.07%)
Class 1: 1117 samples (4.73%)
Class 2: 838 samples (3.54%)
Class 3: 2201 samples (9.31%)
Class 4: 1684 samples (7.12%)
Class 5: 1101 samples (4.66%)
Class 6: 5601 samples (23.69%)
Class 7: 3516 samples (14.87%)

results better but overfitting in epoch 7 some anomaly in epoch2

Using device: mps
Epoch [1/10]
Train Loss: 1.2031, Train Acc: 54.20%
Val Loss: 1.2173, Val Acc: 53.57%
------------------------------------------------------------
Epoch [2/10]
Train Loss: 1.0492, Train Acc: 60.41%
Val Loss: 2.6792, Val Acc: 28.30%
------------------------------------------------------------
Epoch [3/10]
Train Loss: 0.9861, Train Acc: 62.93%
Val Loss: 1.1289, Val Acc: 58.03%
------------------------------------------------------------
Epoch [4/10]
Train Loss: 0.9411, Train Acc: 64.69%
Val Loss: 1.1186, Val Acc: 58.22%
------------------------------------------------------------
Epoch [5/10]
Train Loss: 0.9018, Train Acc: 66.05%
Val Loss: 1.0311, Val Acc: 62.61%
------------------------------------------------------------
Epoch [6/10]
Train Loss: 0.8710, Train Acc: 67.18%
Val Loss: 0.9982, Val Acc: 63.58%
------------------------------------------------------------
Epoch [7/10]
Train Loss: 0.8406, Train Acc: 68.33%
Val Loss: 1.0112, Val Acc: 63.46%

"""


class TissueCNN(nn.Module):
    def __init__(self):
        super(TissueCNN, self).__init__()
        
        # Simpler architecture with fewer layers and no dropout

        # Convulutional layers

        """
            1. Input channel, output channels, kernel size, padding
            2. Normilize output across batch (training stability and speed)
            3. activation function (introduce non linearity, negative values replaced with 0)
            4. Max pooling layer (reduce spatial dimensions, takes max value in each window, helps with computational efficiency and feature invariance)
        """
        self.features = nn.Sequential(
            
            # First block            
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers

        # Adjusted for 64x64 input
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=10, use_wandb=False):
    if use_wandb:
        wandb.init(project="tissue-mnist", name="tissue_classifier")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Simpler training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher learning rate
    criterion = nn.CrossEntropyLoss()
    
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
    
    if use_wandb:
        wandb.finish()
    
    return model

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
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    train_loader, val_loader = prepare_data(batch_size=32)
    
    # Create and train model
    model = TissueCNN()
    trained_model = train_model(model, train_loader, val_loader, use_wandb=True)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'tissue_classifier.pth')