# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import TissueMNIST
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, GuidedGradCam, Saliency
import torchvision.transforms.functional as F


"""
Let's try a completely different approach focusing on the model's learning dynamics:
Focal Loss instead of CrossEntropyLoss to handle class imbalance better
"""


class TissueCNN(nn.Module):
    def __init__(self):
        super(TissueCNN, self).__init__()
        
        # Consistent dropout rate
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

    def get_activation(self, name):
        """Helper method to get intermediate activations"""
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def get_last_conv_layer(self):
        """Helper method to get the last convolutional layer"""
        return self.features[-6]  # Returns the last Conv2d layer before the final MaxPool2d

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

def visualize_explanation(image, attributions, method_names):
    """Helper function to visualize multiple attribution methods"""
    plt.figure(figsize=(15, 3))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # Visualize each attribution method
    for idx, (attribution, method_name) in enumerate(zip(attributions, method_names), start=2):
        plt.subplot(1, 4, idx)
        plt.title(method_name)
        
        # Ensure attribution is detached and converted to numpy
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.detach().cpu().numpy()
        
        # Normalize attribution scores
        attribution = np.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        plt.imshow(attribution.squeeze(), cmap='hot')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def explain_prediction(model, image, target_class, device):
    """
    Generate and visualize different explanations for a model's prediction
    
    Args:
        model: The trained TissueCNN model
        image: Input image tensor (1, 1, 64, 64)
        target_class: The class to explain
        device: The device to run computations on
    """
    model.eval()
    image = image.to(device)
    image.requires_grad = True  # Enable gradients for the input
    
    # Initialize attribution methods
    saliency = Saliency(model)
    guided_gradcam = GuidedGradCam(model, model.get_last_conv_layer())
    ig = IntegratedGradients(model)
    
    # Generate attributions
    saliency_attribution = saliency.attribute(image, target=target_class)
    gradcam_attribution = guided_gradcam.attribute(image, target=target_class)
    ig_attribution = ig.attribute(image, target=target_class, n_steps=50)
    
    # Collect all attributions and method names
    attributions = [
        saliency_attribution,
        gradcam_attribution,
        ig_attribution
    ]
    
    method_names = [
        'Saliency Map',
        'Guided GradCAM',
        'Integrated Gradients'
    ]
    
    # Visualize all explanations
    visualize_explanation(image, attributions, method_names)
    
    # Print model's confidence for this prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][target_class].item()
        print(f"Model's confidence for class {target_class}: {confidence:.2%}")

def analyze_model_decisions(model, test_loader, device, num_samples=5):
    """
    Analyze model decisions on a few test samples
    """
    model.eval()
    samples_analyzed = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if samples_analyzed >= num_samples:
                break
                
            image = images[0].unsqueeze(0)  # Take first image from batch
            true_label = labels[0].item()
            
            # Get model's prediction
            output = model(image.to(device))
            predicted_class = output.argmax(dim=1).item()
            
            print(f"\nAnalyzing sample {samples_analyzed + 1}")
            print(f"True label: {true_label}, Predicted label: {predicted_class}")
            
            # Generate explanations
            explain_prediction(model, image, predicted_class, device)
            samples_analyzed += 1

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
    
    # Add XAI analysis
    print("\nGenerating explanations for model decisions...")
    analyze_model_decisions(model, test_loader, device)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(batch_size=32)
    
    # Create and train model
    model = TissueCNN()
    trained_model = train_model(model, train_loader, val_loader, use_wandb=False)
    
    # Save the trained model's state dict
    torch.save(trained_model.state_dict(), 'tissue_classifier.pth')
    
    # Evaluate model
    evaluate_model(trained_model, test_loader)
    
    # After evaluation, analyze some test samples
    print("\nAnalyzing model decisions...")
    analyze_model_decisions(trained_model, test_loader, device)