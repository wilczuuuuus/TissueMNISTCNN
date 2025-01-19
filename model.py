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
val acc was higher then training 
it changed after 
- remove droput layer
- remove weight decay
- simplified network significantly
- higher learning rate

The higher training accuracy is now more typical because:
Without dropout, the model can fit the training data more easily
Without weight decay, the model can adjust its weights more freely
The higher learning rate allows the model to adapt more quickly to the training data
The simpler architecture has fewer constraints on learning
This is actually a more normal behavior for neural networks - they typically perform better on data they've seen (training) than on data they haven't (validation). The previous behavior (higher val acc) was unusual and likely indicated underfitting due to over-regularization.

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

def prepare_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets with diagnostics
    train_dataset = TissueMNIST(split="train", transform=transform, download=True)
    val_dataset = TissueMNIST(split="val", transform=transform, download=True)
    
    # Print dataset sizes and class distributions
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Calculate and print class distribution
    train_labels = [label.item() for _, label in train_dataset]
    val_labels = [label.item() for _, label in val_dataset]
    
    def print_class_distribution(labels, name):
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"\n{name} class distribution:")
        for class_idx, count in dist.items():
            print(f"Class {class_idx}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    print_class_distribution(train_labels, "Training")
    print_class_distribution(val_labels, "Validation")
    
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
    trained_model = train_model(model, train_loader, val_loader, use_wandb=False)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'tissue_classifier.pth')