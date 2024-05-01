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
import time


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

"""### BINARY INPUT DETECTOR"""

from art.defences.detector.evasion import BinaryInputDetector
from art.estimators.classification import PyTorchClassifier

# Define a simple binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(14*14*16, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        return x

binary_model = BinaryClassifier().to(DEVICE)
binary_criterion = nn.CrossEntropyLoss()
binary_optimizer = torch.optim.Adam(binary_model.parameters(), lr=LEARNING_RATE)

# Wrap the model with ART's classifier
binary_classifier = PyTorchClassifier(
    model=binary_model,
    clip_values=(0, 1),
    loss=binary_criterion,
    optimizer=binary_optimizer,
    input_shape=(1, 28, 28),
    nb_classes=2,
    device_type=DEVICE
)

# Initialize the detector
start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device=DEVICE)
    torch.cuda.empty_cache()  # Clear cache before starting the training
    start_memory = torch.cuda.memory_allocated(device=DEVICE)
detector = BinaryInputDetector(detector=binary_classifier)

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# FGSM and PGD attacks
fgsm_epsilons = [0.2, 0.25, 0.3]
pgd_epsilons = [0.1, 0.15, 0.2]

# Initialize attacks
fgsm_attacks = [FastGradientMethod(estimator=binary_classifier, eps=eps) for eps in fgsm_epsilons]
pgd_attacks = [ProjectedGradientDescent(estimator=binary_classifier, eps=eps) for eps in pgd_epsilons]

# Generate adversarial examples for the batches
adversarial_data = []
clean_data = []
labels = []
batch_counter = 0

for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
    x_batch_np = x_batch.numpy()  # Convert batch to NumPy array for ART processing
    
    # Select the attack based on batch index
    if batch_idx < 3:  # First three batches for FGSM
        attack = fgsm_attacks[batch_idx % len(fgsm_attacks)]
    else:  # Next three batches for PGD
        attack = pgd_attacks[(batch_idx - 3) % len(pgd_attacks)]
    
    # Generate adversarial examples
    adv_examples = attack.generate(x=x_batch_np)
    adversarial_data.append(adv_examples)
    clean_data.append(x_batch_np)
    labels.append(y_batch.numpy())  # Store labels for later use

    # Increment the batch counter
    batch_counter += 1
    if batch_counter >= 6:  # Only process the first six batches
        break

# Flatten lists if necessary
adversarial_data = np.concatenate(adversarial_data)
clean_data = np.concatenate(clean_data)
labels = np.concatenate(labels)

# Combine all data and labels
x_train_detector = np.concatenate([clean_data, adversarial_data])
y_train_detector = np.concatenate([np.zeros(len(clean_data)), np.ones(len(adversarial_data))])

# Assuming detector has been initialized as shown previously
detector.fit(x=x_train_detector, y=y_train_detector, batch_size=128, nb_epochs=10)

# End timer
end_time = time.time()
if torch.cuda.is_available():
    end_memory = torch.cuda.memory_allocated(device=DEVICE)
    max_memory = torch.cuda.max_memory_allocated(device=DEVICE)

def bytes_to_mb(bytes):
    return bytes / (1024 * 1024)

# Calculate the duration
training_duration = end_time - start_time
print(f"Training time for Detector: {training_duration:.2f} seconds.")
print(f"Memory used before training: {bytes_to_mb(start_memory):.2f} MB")
print(f"Memory used after training: {bytes_to_mb(end_memory):.2f} MB")
print(f"Peak memory during training: {bytes_to_mb(max_memory):.2f} MB")

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



"""#EVALAUTION"""
def evaluate_model_with_detector_and_report(model, detector, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    

    all_true_labels = []
    all_pred_labels = []
    total_processed = 0
    total_inference_time = 0

    for batch_idx, (features, labels) in enumerate(data_loader):
        features = features.to(device)
        labels = labels.to(device)

        # Start timing here
        start_time = time.time()

        # Perform detection
        report, is_adversarial = detector.detect(features.cpu().numpy(), batch_size=features.size(0))
        
        # Filter clean samples
        clean_indices = [i for i, is_adv in enumerate(is_adversarial) if not is_adv]

        if len(clean_indices) > 0:
            clean_features = features[clean_indices]
            clean_labels = labels[clean_indices]
            
            # Perform inference only on clean samples
            logits, _ = model(clean_features)
            _, predicted_labels = torch.max(logits, 1)
            
            # Store true and predicted labels for classification report
            all_true_labels.extend(clean_labels.cpu().numpy())
            all_pred_labels.extend(predicted_labels.cpu().numpy())
            total_processed += clean_labels.size(0)
        
        # End timing here
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    # Generate classification report
    if total_processed > 0:
        time_per_thousand = (total_inference_time / total_processed) * 1000
        accuracy = 100. * len(all_pred_labels) / total_processed
        class_report = classification_report(all_true_labels, all_pred_labels, digits=4)
        return accuracy, class_report, time_per_thousand
    else:
        return 0, "No clean samples detected."

# Assuming 'model' and 'detector' are loaded and the model is transferred to the correct device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model with detector on the test data
accuracy, report, time_per_1000 = evaluate_model_with_detector_and_report(model, detector, test_loader, device)
print("Accuracy on clean test samples detected(Binary Input Detector): {:.2f}%".format(accuracy))
print("Classification Report:\n", report)
print(f"Inference speed: {time_per_1000:.2f} milliseconds per 1000 images")

"""## Attacking with FGSM AND PGD"""


def evaluate_model_with_attacks_and_detector(model, detector, attack, data_loader, device):
    model.eval()  # Ensure the model is in evaluation mode
     # Ensure the detector is in evaluation mode

    all_true_labels = []
    all_pred_labels = []
    total_processed = 0
    total_inference_time = 0

    for batch_idx, (features, labels) in enumerate(data_loader):
        features = features.to(device)
        labels = labels.to(device)

        # Generate adversarial examples using FGSM
        adv_examples = attack.generate(x=features.cpu().numpy())
        adv_features = torch.tensor(adv_examples).to(device)

        # Start timing here
        start_time = time.time()

        # Perform detection
        report, is_adversarial = detector.detect(adv_features.cpu().numpy(), batch_size=adv_features.size(0))
        
        # Filter clean samples (those not detected as adversarial)
        clean_indices = [i for i, is_adv in enumerate(is_adversarial) if not is_adv]
        if len(clean_indices) > 0:
            clean_features = adv_features[clean_indices]
            clean_labels = labels[clean_indices]
            
            # Perform inference only on clean samples
            logits, _ = model(clean_features)
            _, predicted_labels = torch.max(logits, 1)
            
            # Store true and predicted labels for classification report
            all_true_labels.extend(clean_labels.cpu().numpy())
            all_pred_labels.extend(predicted_labels.cpu().numpy())
            total_processed += clean_labels.size(0)

        # End timing here
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    # Generate classification report
    if total_processed > 0:
        time_per_thousand = (total_inference_time / total_processed) * 1000
        accuracy = 100. * len(all_pred_labels) / total_processed
        class_report = classification_report(all_true_labels, all_pred_labels, digits=4)
        return accuracy, class_report, time_per_thousand
    else:
        return 0, "No clean samples detected after detection.", 0.0

# Assuming 'model', 'detector', and 'test_loader' are initialized

# Usage of the function with FGSM and PGD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

"""#FAST GRADIENT SIGN METHOD"""

fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.2)

# Evaluate the model with FGSM and detector on the test data
accuracy, report, time_per_1000 = evaluate_model_with_attacks_and_detector(model, detector, fgsm_attack, test_loader, device)
print("Accuracy on clean test samples detected by Binary Input Detector after FGSM attack: {:.2f}%".format(accuracy))
print("Classification Report:\n", report)
print(f"Inference speed on BinaryID - FGSM : {time_per_1000:.2f} milliseconds per 1000 images")


"""#PORJECTED GRADIENT DESCENT"""
# Initialize the PGD attack
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=40)

# Evaluate the model with PGD and detector on the test data
accuracy, report, time_per_1000 = evaluate_model_with_attacks_and_detector(model, detector, pgd_attack, test_loader, device)
print("Accuracy on clean test samples detected by Binary Input Detector after PGD attack: {:.2f}%".format(accuracy))
print("Classification Report:\n", report)
print(f"Inference speed BinaryID - PGD: {time_per_1000:.2f} milliseconds per 1000 images")






