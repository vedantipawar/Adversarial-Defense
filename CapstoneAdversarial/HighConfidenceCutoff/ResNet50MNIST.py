"""#Imports"""

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
from sklearn.metrics import classification_report
from art.defences.postprocessor import HighConfidence


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

"""#Model Settings"""
##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

"""#MNIST Dataset"""
##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

device = torch.device(DEVICE)
torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):
        
        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        
        x = x.to(device)
        y = y.to(device)
        break


##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model

torch.manual_seed(RANDOM_SEED)

model = resnet34(NUM_CLASSES)
model.to(DEVICE)
 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

"""#Training MNIST"""
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


"""#Evaluation"""
from sklearn.metrics import classification_report

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.set_grad_enabled(False): # save memory during inference
    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)

        # Append to lists
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted_labels.cpu().numpy())

# Generate classification report
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
print(classification_report(y_true, y_pred, target_names=target_names))

"""## MODEL WRAPPERS"""
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
    input_shape=(1, 28, 28),
    nb_classes=NUM_CLASSES,
    device_type=DEVICE
)

# Define the attacks
epsilon = 0.1
fgsm_attack = FastGradientMethod(estimator=classifier, eps=epsilon)
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, max_iter=40)



def evaluate_adversarial_attack_with_high_confidence(model, attack, data_loader, device, cut_off):
    # Initialize the HighConfidence postprocessor
    high_confidence_postprocessor = HighConfidence(cutoff=cut_off, apply_fit=False, apply_predict=True)

    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_examples = 0
    predicted_labels = []
    true_labels = []
    total_inference_time = 0

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        start_time = time.time()  # Start timing before generating adversarial examples

        # Generate adversarial examples
        x_test_adv = attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
        labels = labels.to(device)

        # Perform inference on adversarial examples
        logits, probas = model(x_test_adv_torch)

         # Apply High Confidence postprocessing
        probas_detached = probas.detach().cpu().numpy()  # Detach and convert to NumPy array
        probas_postprocessed_np = high_confidence_postprocessor(probas_detached)  # Apply HighConfidence
        probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)  # Convert back to PyTorch tensor

        # Predict labels based on postprocessed probabilities
        _, predicted_labels_batch = torch.max(probas_postprocessed, 1)

        # End timing here
        end_time = time.time()
        total_inference_time += (end_time - start_time)

        # Update the accumulators
        total_correct += (predicted_labels_batch == labels).sum().item()
        total_examples += labels.size(0)
        predicted_labels.extend(predicted_labels_batch.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # Calculate the overall accuracy and generate classification report
    accuracy = total_correct / total_examples * 100 if total_examples > 0 else 0
    class_report = classification_report(true_labels, predicted_labels, target_names=[f"Class {i}" for i in range(10)])
    time_per_thousand = (total_inference_time / total_examples) * 1000 if total_examples > 0 else 0

    return accuracy, class_report, time_per_thousand

# Usage example with FGSM and PGD:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Transfer the model to the appropriate device


for cutoff in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # Evaluate FGSM attack with High Confidence
    accuracy, report, speed = evaluate_adversarial_attack_with_high_confidence(model, fgsm_attack, test_loader, device, cut_off=cutoff)
    print(f"FGSM Attack Accuracy with {cutoff:.2f} High Confidence: {accuracy:.2f}%")


    # Evaluate PGD attack with High Confidence
    accuracy, report, speed = evaluate_adversarial_attack_with_high_confidence(model, pgd_attack, test_loader, device, cutoff)
    print(f"PGD Attack Accuracy with cutoff: {cutoff:.2} High Confidence: {accuracy:.2f}%")

