
## Imports

import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

"""## Model Settings"""

##########################
### SETTINGS
##########################

RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = False



##########################
### CIFAR-10 DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.CIFAR10(root='data', 
                                 train=True, 
                                 transform=transforms.ToTensor(),
                                 download=True)

test_dataset = datasets.CIFAR10(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          num_workers=8,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE,
                         num_workers=8,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas



def resnet34(num_classes):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model

torch.manual_seed(RANDOM_SEED)
model = resnet34(NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize global variables
"""#Training CIFAR"""
# Initialize memory variables
total_memory_allocated = 0
total_memory_cached = 0

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

# Function to save model checkpoint
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

import time
start_time = time.time()  # Track start time
# Load last checkpoint if available
checkpoint_file = 'normalTraining.pth'  # Use direct file path
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
else:
    start_epoch = 0


# Training loop
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Forward pass
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # Logging
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), cost))
    
    # Save model checkpoint
    save_checkpoint(epoch, model, optimizer, checkpoint_file)
    
    # Evaluation on training set
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, 
              compute_accuracy(model, train_loader, device=DEVICE)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    # Memory tracking
    memory_allocated = torch.cuda.memory_allocated(DEVICE)
    memory_cached = torch.cuda.memory_reserved(DEVICE)
    total_memory_allocated += memory_allocated
    total_memory_cached += memory_cached

# Convert bytes to megabytes for readability
total_memory_allocated_mb = total_memory_allocated / (1024 * 1024)
total_memory_cached_mb = total_memory_cached / (1024 * 1024)

print(f'Total Memory Allocated: {total_memory_allocated_mb:.2f} MB')
print(f'Total Memory Cached: {total_memory_cached_mb:.2f} MB')
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

from sklearn.metrics import classification_report

"""## Evaluation"""
import torch

model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []
device = DEVICE
with torch.no_grad():  # Inference mode, no gradients needed
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())  # Collect predictions
        all_labels.extend(targets.cpu().numpy())  # Collect actual labels

# Now `all_preds` and `all_labels` contain all the predictions and labels respectively
print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))



#Importing Adversarial attacks
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)  # Assuming the model returns logits and probabilities
        return logits

model.eval()  # Set the model to evaluation mode


wrapped_model = ModelWrapper(model)

# Wrap the PyTorch model with ART's PyTorchClassifier
classifier = PyTorchClassifier(
    model=wrapped_model,
    clip_values=(0, 1),
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=NUM_CLASSES,
    device_type=DEVICE
)

"""##Evaluaiton with EAE"""
import torch
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import time

def emulated_autoencoder(images):
    resize_down = transforms.Resize(14, interpolation=transforms.InterpolationMode.BILINEAR)  # Explicit bilinear interpolation
    resize_up = transforms.Resize(28, interpolation=transforms.InterpolationMode.BILINEAR)
    return resize_up(resize_down(images))

def test_model_with_eae(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_images = 0
    all_preds = []
    all_targets = []
    start_time = time.time()  # Start timing

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Apply the Emulated Autoencoder
            features = emulated_autoencoder(features)

            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)

            # Accumulate for accuracy
            total_correct += (predicted_labels == targets).sum().item()
            total_images += targets.size(0)

            # Collect all predictions and true labels for classification report
            all_preds.extend(predicted_labels.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).cpu().numpy())


    end_time = time.time()  # End timing
    elapsed_time = ((end_time - start_time) / total_images) * 1000
    accuracy = (total_correct / total_images) * 100

    # Output classification report
    print(f'Classification Report EAE:\n{classification_report(all_targets, all_preds)}')
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Time required to process {total_images} images: {elapsed_time:.2f} seconds')

    return accuracy, elapsed_time

# Use this modified function to test your model
device = torch.device(DEVICE)
model.to(device)
accuracy, test_time = test_model_with_eae(model, test_loader, device)

"""#EAE AGAINST ATTACKS"""
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

import torch
import torchvision.transforms as transforms
from sklearn.metrics import classification_report


# Adjusted function to compute accuracy on FGSM with EAE
def compute_accuracy_with_fgsm_and_eae(model, fgsm_attack, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels, predicted_labels = [], []

    model.eval()
    start_time = time.time()

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        # Generate FGSM adversarial examples
        x_test_adv = fgsm_attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)

        # Apply Emulated Autoencoder to the adversarial examples
        x_test_adv_torch = emulated_autoencoder(x_test_adv_torch)

        labels = labels.to(device)

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)
            predicted_labels_batch = torch.argmax(probas, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)
        
        end_time = time.time()
    elapsed_time = ((end_time - start_time)/ num_examples) * 1000
    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on FGSM adversarial examples after EAE: {accuracy:.2f}%')
    print(f'Time required to process first 1000 images: {elapsed_time:.2f} seconds')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.2)
compute_accuracy_with_fgsm_and_eae(model, fgsm_attack, test_loader, DEVICE)


# Adjusted function to compute accuracy on PGD with EAE
# Initialize the PGD attack
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=40)
def compute_accuracy_with_pgd_and_eae(model, pgd_attack, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels, predicted_labels = [], []
    total_processed = 0
    total_inference_time = 0

    model.eval()
    start_time = time.time()

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        # Generate PGD adversarial examples
        x_test_adv = pgd_attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)

        # Apply Emulated Autoencoder to the adversarial examples
        x_test_adv_torch = emulated_autoencoder(x_test_adv_torch)

        labels = labels.to(device)

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)
            predicted_labels_batch = torch.argmax(probas, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

        end_time= time.time()
            
    elapsed_time = ((end_time - start_time)/ num_examples) * 1000
    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on PGD adversarial examples after EAE: {accuracy:.2f}%')
    print(f'Time required to process first 1000 images: {elapsed_time:.2f} seconds')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

compute_accuracy_with_pgd_and_eae(model, pgd_attack, test_loader, DEVICE)

