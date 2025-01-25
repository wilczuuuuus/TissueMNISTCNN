import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import TissueMNIST
import os

class PretrainedTissueCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(PretrainedTissueCNN, self).__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Modify first conv layer to accept 1 channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        # Squeeze the target tensor to make it 1D
        target = target.squeeze()
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# Update the transforms to match ImageNet preprocessing
base_transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # ImageNet stats for grayscale
])

augment_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# ... keep AugmentedTissueMNIST class unchanged ...

def prepare_data(batch_size=32, augment=True):
    """
    Prepare train, validation and test dataloaders
    """
    # Create data directory if it doesn't exist
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = TissueMNIST(root=data_dir, split="train", download=True, 
                               transform=augment_transform if augment else base_transform)
    val_dataset = TissueMNIST(root=data_dir, split="val", download=True, 
                             transform=base_transform)
    test_dataset = TissueMNIST(root=data_dir, split="test", download=True, 
                              transform=base_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=30, use_wandb=False, patience=5):
    if use_wandb:
        wandb.init(project="tissue-mnist", name="pretrained_tissue_classifier")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Modified learning rates and optimizer parameters
    params = [
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': 5e-5},
        {'params': model.backbone.fc.parameters(), 'lr': 5e-4}
    ]
    
    optimizer = optim.AdamW(params, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = FocalLoss(gamma=1.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.squeeze()  # Add this line to handle dimension
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = target.squeeze()  # Add this line to handle dimension
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total
        
        # Print progress
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')
        
        if use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if use_wandb:
        wandb.finish()
    
    return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(batch_size=64)
    
    # Create and train model
    model = PretrainedTissueCNN()
    trained_model = train_model(model, train_loader, val_loader, num_epochs=30, use_wandb=True)
    
    # Save the trained model's state dict
    torch.save(trained_model.state_dict(), 'pretrained_tissue_classifier.pth')