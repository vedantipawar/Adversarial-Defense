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

"""#Training MNIST with CUTMIX"""
from art.defences.preprocessor import CutMixPyTorch

cutmix = CutMixPyTorch(num_classes=NUM_CLASSES, alpha=1.0, probability=0.5, channels_first=True, apply_fit=True, apply_predict=False, device_type='cuda', verbose=False)

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
checkpoint_file = 'cutMixTraining.pth'  # Use direct file path
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
else:
    start_epoch = 0


# Training loop
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix augmentation
        inputs, targets = cutmix.forward(inputs, targets)

        # Forward pass
        outputs, _ = model(inputs)  # Adjust this line if your model returns logits and probas
        # If your model directly returns a tuple, make sure to select the logits part here

        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, len(train_loader), loss))
    
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

"""#Attacks without any defense"""

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
    print(f'Accuracy on FGSM adversarial examples with: {accuracy:.2f}%')
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
    print(f'Accuracy on PGD adversarial examples with CUTMIX: {accuracy:.2f}%')
    print(f'Time required to process first 1000 images: {elapsed_time:.2f} seconds')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

compute_accuracy_with_pgd_and_eae(model, pgd_attack, test_loader, DEVICE)



"""#Post Processing Techniques"""

##############################
### GAUSSIAN NOISE #######
#############################

from art.defences.postprocessor import GaussianNoise

# Initialize the GaussianNoise postprocessor
gaussian_noise_postprocessor = GaussianNoise(scale=0.2, apply_fit=False, apply_predict=True)


# On CLEAN IMAGES
# Variables to hold metrics
total_correct = 0
total_examples = 0
predicted_labels = []
true_labels = []

# Ensure the model is in evaluation mode
model.eval()

for images, labels in test_loader:
    # Convert images to NumPy array for ART
    images_np = images.numpy()

    # Apply Gaussian Noise to clean examples
    images_np_noised = gaussian_noise_postprocessor(images_np)

    # Convert noised images back to PyTorch tensors and move to the correct device
    images_noised_torch = torch.from_numpy(images_np_noised).to(DEVICE)
    labels = labels.to(DEVICE)

    # Perform inference on noised examples
    logits, _ = model(images_noised_torch)
    _, predictions = torch.max(logits, dim=1)

    # Update the accumulators
    total_correct += (predictions == labels).sum().item()
    total_examples += labels.size(0)

    # Accumulate predicted and true labels for later analysis
    predicted_labels.extend(predictions.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Calculate the overall accuracy
accuracy_noised = total_correct / total_examples
print(f"Accuracy on clean examples with Gaussian Noise: {accuracy_noised * 100:.2f}%")

# Generate classification report
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
print(classification_report(true_labels, predicted_labels, target_names=target_names))

#FAST GRADIENT SIGN METHOD
# Create FGSM attack
fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.2)
from sklearn.metrics import classification_report

total_correct = 0
total_examples = 0
predicted_labels = []
true_labels = []

# Ensure the model is in evaluation mode
model.eval()

for images, labels in test_loader:
    # Convert images to NumPy array for ART
    images_np = images.numpy()

    # Generate adversarial examples
    x_test_adv = fgsm_attack.generate(x=images_np)

    # Apply Gaussian Noise to adversarial examples
    x_test_adv_noised = gaussian_noise_postprocessor(x_test_adv)

    # Convert noised adversarial examples back to PyTorch tensors and move to the correct device
    x_test_adv_noised_torch = torch.from_numpy(x_test_adv_noised).to(DEVICE)
    labels = labels.to(DEVICE)

    # Perform inference on noised adversarial examples
    logits, _ = model(x_test_adv_noised_torch)
    _, predictions = torch.max(logits, dim=1)

    # Update the accumulators
    total_correct += (predictions == labels).sum().item()
    total_examples += labels.size(0)

    # Accumulate predicted and true labels
    predicted_labels.extend(predictions.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Calculate the overall accuracy
accuracy_fgsm = total_correct / total_examples
print(f"Accuracy on FGSM adversarial examples defense: CUTMIX - Gaussian Noise: {accuracy_fgsm * 100:.2f}%")

# Generate classification report
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
print(classification_report(true_labels, predicted_labels, target_names=target_names))

"""#PROJECT GRADIENT DESCENT"""

# Create PGD attack
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=40)

total_correct = 0
total_examples = 0
predicted_labels = []
true_labels = []

# Create PGD attack

# Ensure the model is in evaluation mode
model.eval()

for images, labels in test_loader:
    # Convert images to NumPy array for ART
    images_np = images.numpy()

    # Generate adversarial examples using PGD attack
    x_test_adv = pgd_attack.generate(x=images_np)

    # Apply Gaussian Noise to adversarial examples
    x_test_adv_noised = gaussian_noise_postprocessor(x_test_adv)

    # Convert noised adversarial examples back to PyTorch tensors and move to the correct device
    x_test_adv_noised_torch = torch.from_numpy(x_test_adv_noised).to(DEVICE)
    labels = labels.to(DEVICE)

    # Perform inference on noised adversarial examples
    logits, _ = model(x_test_adv_noised_torch)
    _, predictions = torch.max(logits, dim=1)

    # Update the accumulators
    total_correct += (predictions == labels).sum().item()
    total_examples += labels.size(0)

    # Accumulate predicted and true labels
    predicted_labels.extend(predictions.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Calculate the overall accuracy
accuracy_pgd = total_correct / total_examples
print(f"Accuracy on PGD adversarial examples defense: CUTMIX - Gaussian Noise: {accuracy_pgd * 100:.2f}%")

# Generate classification report
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
print(classification_report(true_labels, predicted_labels, target_names=target_names))


##############################
### HIGH CONFIDENCE #######
#############################
from art.defences.postprocessor import HighConfidence

high_confidence_postprocessor = HighConfidence(cutoff=0.50, apply_fit=False, apply_predict=True)


"""#ON CLEAN IMAGES"""
from sklearn.metrics import classification_report
import torch

def compute_accuracy_with_high_confidence_on_clean_images(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    model.eval()

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            logits, probas = model(images)

            # Apply High Confidence postprocessing
            probas_np = probas.cpu().numpy()  # Convert to NumPy array for ART
            probas_postprocessed_np = high_confidence_postprocessor(probas_np)  # Apply HighConfidence
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)  # Convert back to PyTorch tensor

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy with High Confidence postprocessing on clean images: {accuracy:.2f}%')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names, digits=4))

# Usage of the function
compute_accuracy_with_high_confidence_on_clean_images(model, test_loader, DEVICE)



"""# Fast Gradient Sign Method"""

def compute_accuracy_with_high_confidence_on_fgsm(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    model.eval()

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        # Generate FGSM adversarial examples
        x_test_adv = fgsm_attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)

            # Apply High Confidence postprocessing
            probas_np = probas.cpu().numpy()  # Convert to NumPy array for ART
            probas_postprocessed_np = high_confidence_postprocessor(probas_np)  # Apply HighConfidence
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)  # Convert back to PyTorch tensor

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on FGSM adversarial examples defense: CUTMIX - High Confidence postprocessing: {accuracy:.2f}%')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

compute_accuracy_with_high_confidence_on_fgsm(model, test_loader, DEVICE)

"""# Project Gradient Descent"""

from sklearn.metrics import classification_report

def compute_accuracy_with_high_confidence_on_pgd(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    model.eval()

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        # Generate PGD adversarial examples
        x_test_adv = pgd_attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)

            # Apply High Confidence postprocessing
            probas_np = probas.cpu().numpy()  # Convert to NumPy array for ART
            probas_postprocessed_np = high_confidence_postprocessor(probas_np)  # Apply HighConfidence
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)  # Convert back to PyTorch tensor

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on PGD adversarial examples defense: CUTMIX - High Confidence postprocessing: {accuracy:.2f}%')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

compute_accuracy_with_high_confidence_on_pgd(model, test_loader, DEVICE)

##############################
#### REVERSE SIGMOID #######
#############################

from art.defences.postprocessor import ReverseSigmoid

reverse_sigmoid_postprocessor = ReverseSigmoid(beta=1.0, gamma=0.1, apply_fit=False, apply_predict=True)

#Get Model predictions
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Initialize a list to store the probabilities
probabilities = []

with torch.no_grad():  # Inference without gradient calculation
    for images, _ in test_loader:  # Use the test_loader you defined
        images = images.to(device)  # Move images to the configured device
        _, probas = model(images)  # Get logits and probabilities from your model
        probabilities.extend(probas.cpu().numpy())  # Store probabilities

# Convert the list of probabilities to a NumPy array
probabilities = np.array(probabilities)

# Apply ReverseSigmoid postprocessing
postprocessed_predictions = reverse_sigmoid_postprocessor(probabilities)
# Assuming postprocessed_predictions is a numpy array of shape (num_samples, num_classes)
postprocessed_labels = np.argmax(postprocessed_predictions, axis=1)

ground_truth = []

for _, labels in test_loader:
    ground_truth.extend(labels.numpy())

ground_truth = np.array(ground_truth)


"""#ON CLEAN IMAGES"""


def evaluate_with_reverse_sigmoid(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []
    correct_pred, num_examples = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            _, probas = model(images)  # Get the logits and probabilities from your model
            
            # Convert probabilities to NumPy for postprocessing
            probas_np = probas.cpu().numpy()

            # Apply Reverse Sigmoid postprocessing
            probas_postprocessed_np = reverse_sigmoid_postprocessor(probas_np)
            
            # Convert postprocessed probabilities back to PyTorch tensor and move to the correct device
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

    # Calculate and print accuracy
    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on CLean Test IMages with Reverse Sigmoid postprocessing: {accuracy:.2f}%')

    # Generate and print classification report
    print(classification_report(true_labels, predicted_labels, digits=4))

# Assuming you have defined test_loader and DEVICE
evaluate_with_reverse_sigmoid(model, test_loader, DEVICE)


"""#Fast Gradient Sign Method"""
from sklearn.metrics import classification_report

def compute_accuracy_with_reverse_sigmoid_on_fgsm(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    model.eval()

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        # Generate FGSM adversarial examples
        x_test_adv = fgsm_attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)

            # Apply Reverse Sigmoid postprocessing
            probas_np = probas.cpu().numpy()  # Convert to NumPy array for ART
            probas_postprocessed_np = reverse_sigmoid_postprocessor(probas_np)  # Apply ReverseSigmoid
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)  # Convert back to PyTorch tensor

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on FGSM adversarial examples defense: CUTMIX - Reverse Sigmod: {accuracy:.2f}%')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

compute_accuracy_with_reverse_sigmoid_on_fgsm(model, test_loader, DEVICE)

from sklearn.metrics import classification_report

def compute_accuracy_with_reverse_sigmoid_on_pgd(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    model.eval()

    for images, labels in data_loader:
        images_np = images.numpy()  # Convert images to NumPy array for ART

        # Generate PGD adversarial examples
        x_test_adv = pgd_attack.generate(x=images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)

            # Apply Reverse Sigmoid postprocessing
            probas_np = probas.cpu().numpy()  # Convert to NumPy array for ART
            probas_postprocessed_np = reverse_sigmoid_postprocessor(probas_np)  # Apply ReverseSigmoid
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)  # Convert back to PyTorch tensor

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true labels and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy on PGD adversarial examples defense: CUTMIX - Reverse Sigmoid : {accuracy:.2f}%')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

compute_accuracy_with_reverse_sigmoid_on_pgd(model, test_loader, DEVICE)


