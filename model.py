# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import TissueMNIST


class TissueCNN(nn.Module):
    """
    CNN Model
    - 3 convolutional layers
    - 2 fully connected layers (Linear)
    """

    def __init__(self):
        super(TissueCNN, self).__init__()

        # Convolutional layers:
        
        self.features = nn.Sequential(


            # input channel, output channels, kernel size, padding
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # Normilize output across batch (training stability and speed)
            nn.BatchNorm2d(32),
            # activation function (introduce non linearity, negative values replaced with 0)
            nn.ReLU(),
            # Max pooling layer (reduce spatial dimensions, takes max value in each window, helps with computational efficiency and feature invariance)
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        
        self.classifier = nn.Sequential(
            # Randomly drop out 50% of neurons to prevent overfitting
            nn.Dropout(0.5),

            # Flatten the output of the convolutional layers
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Output layer (8 classes)
            nn.Linear(512, 8)
        )
        
    def forward(self, x):
        # Pass input through convolutional layers
        x = self.features(x)
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
        x = self.classifier(x)
        return x

# Data preparation
def prepare_data(batch_size=32):
    # Simple transforms - just convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load datasets
    train_dataset = TissueMNIST(
        split="train",
        transform=transform,
        download=True,
        size=64
    )
    
    val_dataset = TissueMNIST(
        split="val",
        transform=transform,
        download=True,
        size=64
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, use_wandb=False):
    if use_wandb:
        wandb.init(project="tissue-classification", name="tissue_cnn_run")
        # Log model architecture
        wandb.watch(model)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.squeeze().to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
    
    if use_wandb:
        wandb.finish()
    
    return model


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