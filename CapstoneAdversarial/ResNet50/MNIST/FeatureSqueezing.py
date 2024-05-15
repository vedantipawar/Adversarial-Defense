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
from art.defences.preprocessor import FeatureSqueezing

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

"""#Training MNIST with FEATURE SQUEEZING"""

# Initialize Feature Squeezing
feature_squeezing = FeatureSqueezing(clip_values=(0, 1), bit_depth=4)

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
checkpoint_file = 'FeatureSqueezingTraining.pth'  # Use direct file path
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

        # Convert inputs to numpy arrays for feature squeezing
        inputs_np = inputs.cpu().numpy()

        # Apply Feature Squeezing
        squeezed_inputs_np, _ = feature_squeezing(inputs_np)

        # Convert squeezed inputs back to PyTorch tensors and move to the correct device
        squeezed_inputs = torch.tensor(squeezed_inputs_np).to(device)

        # Forward pass
        outputs, _ = model(squeezed_inputs)  # Ensure this line matches your model's output

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
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
print("Classification report after Training using Feature Squeezing: ")
print(classification_report(y_true, y_pred, target_names=target_names))



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

# """# No defense attacks"""
from sklearn.metrics import classification_report, accuracy_score

# """# No defense attacks"""
from sklearn.metrics import classification_report, accuracy_score

# Define FGSM and PGD attacks
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.1)
pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=40)
def evaluate_adversarial_attack(model, attack, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_examples = 0
    predicted_labels = []
    true_labels = []
    total_inference_time = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        start_time = time.time()  # Start timing before generating adversarial examples

         # Squeeze features before adversarial attack generation
        images_np = images.cpu().numpy()
        squeezed_images_np, _ = feature_squeezing(images_np)
        squeezed_images = torch.from_numpy(squeezed_images_np).float().to(device)
        # Generate adversarial examples
        x_test_adv = attack.generate(x=squeezed_images.cpu().detach().numpy())

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
 
        # Perform inference on adversarial examples
        logits, _ = model(x_test_adv_torch)
        _, predictions = torch.max(logits, 1)
        end_time = time.time()  # End timing after predictions

        # Measure inference time
        total_inference_time += (end_time - start_time)

        # Update the accumulators
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)
        predicted_labels.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # Calculate the overall accuracy
    accuracy = (total_correct / total_examples * 100) if total_examples > 0 else 0
    class_report = classification_report(true_labels, predicted_labels, target_names=[f"Class {i}" for i in range(10)])

    # Calculate and print inference speed per 1000 images
    time_per_thousand = (total_inference_time / total_examples) * 1000 if total_examples > 0 else 0

    return accuracy, class_report, time_per_thousand

# Usage example with FGSM and PGD:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Transfer the model to the appropriate device



# Evaluate FGSM attack
accuracy_fgsm, report_fgsm, speed_fgsm = evaluate_adversarial_attack(model, fgsm_attack, test_loader, device)
print(f"FGSM Attack Accuracy: {accuracy_fgsm:.2f}%")
print("FGSM Classification Report:\n", report_fgsm)
print(f"FGSM Inference Speed: {speed_fgsm:.2f} ms per 1000 images")

# Evaluate PGD attack
accuracy_pgd, report_pgd, speed_pgd = evaluate_adversarial_attack(model, pgd_attack, test_loader, device)
print(f"PGD Attack Accuracy: {accuracy_pgd:.2f}%")
print("PGD Classification Report:\n", report_pgd)
print(f"PGD Inference Speed: {speed_pgd:.2f} ms per 1000 images")


# """#Post Processing Techniques"""

# ##############################
# ### GAUSSIAN NOISE #######
# #############################

import torch
import numpy as np
from sklearn.metrics import classification_report
from art.defences.postprocessor import GaussianNoise
from time import time

#Accuracy without attacks:
from art.defences.postprocessor import GaussianNoise

# Initialize the GaussianNoise postprocessor
gaussian_noise_postprocessor = GaussianNoise(scale=0.1, apply_fit=False, apply_predict=True)

model.eval()  # Ensure the model is in evaluation mode
y_true = []
y_pred = []
total_inference_time = 0  # Initialize variable to measure inference time

import time
from sklearn.metrics import classification_report, accuracy_score

with torch.set_grad_enabled(False):  # Save memory during inference
    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # Start timing here
        start_time = time.time()

        #Convert inputs to numpy arrays for feature Squeezing
        features_np = features.cpu().numpy()
        squeezed_features_np, _ = feature_squeezing(features_np)

        # Convert squeezed features back to tensor and pass to device
        squeezed_features = torch.from_numpy(squeezed_features_np).float().to(device)

         # Forward pass and extract logits explicitly if model returns multiple outputs
        outputs = model(squeezed_features)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]  # assuming logits are the first element
        probas = torch.softmax(logits, dim=1)  # Convert logits to probabilities

        logits_np = logits.cpu().numpy()  # Convert logits to numpy array for Gaussian noise application
        noisy_logits_np = gaussian_noise_postprocessor(logits_np)  # Apply Gaussian noise to logits
        noisy_logits = torch.from_numpy(noisy_logits_np).float().to(DEVICE)  # Convert back to tensor and ensure type float

        # Convert noisy logits back to probabilities for classification
        noisy_probas = torch.softmax(noisy_logits, dim=1)
        _, predicted_labels = torch.max(noisy_probas, 1)


        # End timing here
        end_time = time.time()
        total_inference_time += (end_time - start_time)

        # Append to lists for evaluation
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted_labels.cpu().numpy())

# Generate classification report
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
accuracy = accuracy_score(y_true, y_pred)

classification_report_str = classification_report(y_true, y_pred, target_names=target_names)
print(classification_report_str)

# Calculate and print inference speed per 1000 images
total_images = len(y_true)
time_per_thousand = (total_inference_time / total_images) * 1000 if total_images > 0 else 0
print(f"Inference speed on Gaussian without any attack: {time_per_thousand:.2f} seconds per 1000 images")


#Attacks

def evaluate_attack_with_gaussian_noise_defense(model, attack, data_loader, device):
 
    model.eval()  # Ensure the model is in evaluation mode
    total_correct = 0
    total_examples = 0
    predicted_labels = []
    true_labels = []
    inference_time = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)  # Convert images to NumPy array for ART

        start_time = time.time()
        images_np = images.cpu().numpy()
        squeezed_images_np, _= feature_squeezing(images_np)
        squeezed_images = torch.from_numpy(squeezed_images_np).float().to(device)
        
        # Generate adversarial examples
        x_test_adv = attack.generate(x=squeezed_images.cpu().detach().numpy())  # ART needs numpy arrays

        # Convert adversarial examples back to tensors and perform inference
        x_test_adv_torch = torch.from_numpy(x_test_adv).float().to(device)
       # Perform inference and extract logits
        outputs = model(x_test_adv_torch)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        # Apply Gaussian Noise to the output (logits or probabilities)
        logits_noised_np = gaussian_noise_postprocessor(logits.cpu().detach().numpy())  # Apply postprocessing
        logits_noised = torch.from_numpy(logits_noised_np).float().to(device)

        # Convert noised logits to probabilities and get predictions
        probas_noised = torch.softmax(logits_noised, dim=1)
        _, predictions = torch.max(probas_noised, dim=1)
        inference_time += time.time() - start_time

        # Update the accumulators
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)
        predicted_labels.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # Calculate the overall accuracy and generate classification report
    accuracy = total_correct / total_examples * 100 if total_examples > 0 else 0
    class_report = classification_report(true_labels, predicted_labels, target_names=[f"Class {i}" for i in range(10)])
    time_per_thousand = (inference_time / total_examples) * 1000 if total_examples > 0 else 0
    accuracyscore = accuracy_score(true_labels, predicted_labels)
    return accuracy, accuracyscore,class_report, time_per_thousand


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Transfer the model to the appropriate device


# Evaluate FGSM attack with Gaussian Noise
accuracy, accuracyscore,report, speed = evaluate_attack_with_gaussian_noise_defense(model, fgsm_attack, test_loader, device)
print(f"FGSM Attack Accuracy with Gaussian Noise: {accuracy:.2f}%")
print("FGSM Classification Report with Gaussian Noise:\n", report)
print(f"FGSM Inference Speed: {speed:.2f} s per 1000 images")
print(f"Accuracy Score: {accuracyscore:.2f}%")

# Evaluate PGD attack with Gaussian Noise
accuracy, accuracyscore, report, speed = evaluate_attack_with_gaussian_noise_defense(model, pgd_attack, test_loader, device)
print(f"PGD Attack Accuracy with Gaussian Noise: {accuracy:.2f}%")
print("PGD Classification Report with Gaussian Noise:\n", report)
print(f"PGD Inference Speed: {speed:.2f} s per 1000 images")
print(f"Accuracy Score: {accuracyscore:.2f}%")



##############################
### HIGH CONFIDENCE #######
#############################
import torch
from sklearn.metrics import classification_report

"""# High confidence without attacks"""
from art.defences.postprocessor import HighConfidence

# Initialize the HighConfidence postprocessor
high_confidence_postprocessor = HighConfidence(cutoff=0.1, apply_fit=False, apply_predict=True)

model.eval()  # Set the model to evaluation mode
y_true = []
y_pred = []
y_pred_confident = []

with torch.set_grad_enabled(False):  # Disable gradients for inference efficiency
    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        images_np = features.cpu().numpy()
        squeezed_images_np, _ = feature_squeezing(images_np)

        # Convert squeezed images back to tensor and send to the correct device
        squeezed_features = torch.from_numpy(squeezed_images_np).float().to(device)
        
        # Perform inference
        outputs = model(squeezed_features)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]  # Extract logits if model returns a tuple


        probas = torch.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)


        # Apply high confidence postprocessing
        confident_probas = high_confidence_postprocessor(probas.cpu().numpy())
        _, confident_labels = torch.max(torch.tensor(confident_probas), 1)

        # Append original predictions
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted_labels.cpu().numpy())

        # Append high confidence predictions
        y_pred_confident.extend(confident_labels.numpy())

accuracyscore = accuracy_score(y_true, y_pred)
# Generate and print classification reports
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]

print("High Confidence Classification Report and accuracy:")
print(f"Accuracy: {accuracyscore:.2f}%")
print(classification_report(y_true, y_pred_confident, target_names=target_names, zero_division=0))



"""# Attacks"""
def evaluate_adversarial_attack_with_high_confidence(model, attack, data_loader, device):

    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_examples = 0
    predicted_labels = []
    true_labels = []
    total_inference_time = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)  # Convert images to NumPy array for ART

        
        start_time = time.time()  # Start timing before generating adversarial examples

        # Convert images to NumPy array for ART after moving them to CPU
        images_np = images.cpu().numpy()
        squeezed_images_np, _ = feature_squeezing(images_np)
        squeezed_images = torch.from_numpy(squeezed_images_np).float().to(device)

        # Generate adversarial examples
        x_test_adv = attack.generate(x=squeezed_images.cpu().detach().numpy())

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).float().to(device)
        
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

# Evaluate FGSM attack with High Confidence
accuracy, report, speed = evaluate_adversarial_attack_with_high_confidence(model, fgsm_attack, test_loader, device)
print(f"FGSM Attack Accuracy with High Confidence: {accuracy:.2f}%")
print("FGSM Classification Report with High Confidence:\n", report)
print(f"FGSM Inference Speed: {speed:.2f} s per 1000 images")

# Evaluate PGD attack with High Confidence
accuracy, report, speed = evaluate_adversarial_attack_with_high_confidence(model, pgd_attack, test_loader, device)
print(f"PGD Attack Accuracy with High Confidence: {accuracy:.2f}%")
print("PGD Classification Report with High Confidence:\n", report)
print(f"PGD Inference Speed: {speed:.2f} s per 1000 images")


##############################
### GAUSSIAN NOISE AS NOISE and not after output prediction #######
#############################

from art.defences.postprocessor import GaussianNoise

# Initialize the GaussianNoise postprocessor
gaussian_noise_postprocessor = GaussianNoise(scale=0.1, apply_fit=False, apply_predict=True)

model.eval()  # Ensure the model is in evaluation mode
y_true = []
y_pred = []
total_inference_time = 0  # Initialize variable to measure inference time

import time
from sklearn.metrics import classification_report

with torch.set_grad_enabled(False):  # Save memory during inference
    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # Start timing here
        start_time = time.time()

        # Apply Gaussian noise
        features_np = features.cpu().numpy()  # Convert to numpy array for Gaussian noise application
        noised_features_np = gaussian_noise_postprocessor(features_np)  # Apply Gaussian noise
        noised_features = torch.from_numpy(noised_features_np).to(DEVICE)  # Convert back to tensor and send to device

        # Forward pass with noised data
        logits, probas = model(noised_features)
        _, predicted_labels = torch.max(probas, 1)

        # End timing here
        end_time = time.time()
        total_inference_time += (end_time - start_time)

        # Append to lists for evaluation
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted_labels.cpu().numpy())

# Generate classification report
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
classification_report_str = classification_report(y_true, y_pred, target_names=target_names)
print('Accuracy on clean test images with FeatureSqueezing training and Gaussian noise preprocessor for testing: ')
print(classification_report_str)

# Calculate and print inference speed per 1000 images
total_images = len(y_true)
time_per_thousand = (total_inference_time / total_images) * 1000 if total_images > 0 else 0
print(f"Inference speed on FeatureSqueezing - Gaussian noise preprocessor for testing without any attack: {time_per_thousand:.2f} seconds per 1000 images")


#Attacks

def evaluate_attack_with_gaussian_noise_defense(model, attack, data_loader, device):
    # Initialize the GaussianNoise postprocessor

    model.eval()  # Ensure the model is in evaluation mode
    total_correct = 0
    total_examples = 0
    predicted_labels = []
    true_labels = []
    inference_time = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        start_time = time.time()
        # Convert images to NumPy array for ART
        images_np = images.cpu().numpy()

        # Apply feature squeezing
        squeezed_images_np, _ = feature_squeezing(images_np)

        # Generate adversarial examples
        x_test_adv = attack.generate(x=squeezed_images_np)

        # Apply Gaussian Noise to adversarial examples
        x_test_adv_noised = gaussian_noise_postprocessor(x_test_adv)

        # Convert noised adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_noised_torch = torch.from_numpy(x_test_adv_noised).to(device)
        labels = labels.to(device)

        # Perform inference on noised adversarial examples
        logits, _ = model(x_test_adv_noised_torch)
        _, predictions = torch.max(logits, dim=1)
        inference_time += time.time() - start_time

        # Update the accumulators
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)
        predicted_labels.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    # Calculate the overall accuracy and generate classification report
    accuracy = total_correct / total_examples * 100 if total_examples > 0 else 0
    class_report = classification_report(true_labels, predicted_labels, target_names=[f"Class {i}" for i in range(10)])
    time_per_thousand = (inference_time / total_examples) * 1000 if total_examples > 0 else 0

    return accuracy, class_report, time_per_thousand

# Example usage with FGSM and PGD
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Transfer the model to the appropriate device


# Evaluate FGSM attack with Gaussian Noise
accuracy, report, speed = evaluate_attack_with_gaussian_noise_defense(model, fgsm_attack, test_loader, device)
print(f"FGSM Attack Accuracy with FeatureSqueezing - Gaussian Noise as pretest: {accuracy:.2f}%")
print("FGSM Classification Report with FeatureSqueezing - Gaussian Noise as pretest:\n", report)
print(f"FGSM Inference Speed: {speed:.2f} ms per 1000 images")

# Evaluate PGD attack with Gaussian Noise
accuracy, report, speed = evaluate_attack_with_gaussian_noise_defense(model, pgd_attack, test_loader, device)
print(f"PGD Attack Accuracy with FeatureSqueezing - Gaussian Noise as pretest: {accuracy:.2f}%")
print("PGD Classification Report with FeatureSqueezing - Gaussian Noise as pretest:\n", report)
print(f"PGD Inference Speed: {speed:.2f} ms per 1000 images")

##############################
#### REVERSE SIGMOID #######
#############################

from art.defences.postprocessor import ReverseSigmoid
reverse_sigmoid_postprocessor = ReverseSigmoid(beta=1.0, gamma=0.1, apply_fit=False, apply_predict=True)

"""#Accuracy on clean images RS without attacks"""
def evaluate_with_reverse_sigmoid(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []
    correct_pred, num_examples = 0, 0
    total_inference_time = 0
    with torch.no_grad():
        for images, labels in data_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()

            features_np = images.cpu().numpy()
            squeezed_features_np, _ = feature_squeezing(features_np)
            squeezed_features = torch.from_numpy(squeezed_features_np).to(device)

            _, probas = model(squeezed_features)  # Get the logits and probabilities from your model
            
            # Convert probabilities to NumPy for postprocessing
            probas_np = probas.cpu().numpy()

            # Apply Reverse Sigmoid postprocessing
            probas_postprocessed_np = reverse_sigmoid_postprocessor(probas_np)
            
            # Convert postprocessed probabilities back to PyTorch tensor and move to the correct device
            probas_postprocessed = torch.from_numpy(probas_postprocessed_np).float().to(device)

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, dim=1)

            # End timing here
            end_time = time.time()
            total_inference_time += (end_time - start_time)
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
    time_per_thousand = (total_inference_time / num_examples) * 1000 if num_examples > 0 else 0
    print('Inference time / 100 images: ', time_per_thousand)
# Assuming you have defined test_loader and DEVICE
evaluate_with_reverse_sigmoid(model, test_loader, DEVICE)


"""#on attacks"""
from art.defences.postprocessor import ReverseSigmoid

def evaluate_model_with_reverse_sigmoid_and_attack(model, data_loader, device, attack, reverse_sigmoid_params={'beta': 1.0, 'gamma': 0.1}):

    model.eval()  # Set the model to evaluation mode
    correct_pred, num_examples = 0, 0
    true_labels = []
    predicted_labels = []
    total_inference_time = 0  # Initialize total inference time


    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

         # Convert images to NumPy array for ART and apply feature squeezing
        images_np = images.cpu().numpy()
        squeezed_images_np, _ = feature_squeezing(images_np)  # Apply feature squeezing


        # Generate adversarial examples
        x_test_adv = attack.generate(x=squeezed_images_np)

        # Convert adversarial examples back to PyTorch tensors and move to the correct device
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device)
        

        start_time = time.time()  # Start timing before model processing

        with torch.no_grad():
            logits, probas = model(x_test_adv_torch)

            # Apply Reverse Sigmoid postprocessing
            probas_np = probas.cpu().numpy()  # Convert to NumPy array for ART
            probas_postprocessed_np = reverse_sigmoid_postprocessor(probas_np)
            probas_postprocessed = torch.tensor(probas_postprocessed_np).to(device)

            # Predict labels based on postprocessed probabilities
            predicted_labels_batch = torch.argmax(probas_postprocessed, 1)

            # Accumulate true and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_labels_batch.cpu().numpy())

            # Calculate correct predictions
            correct_pred += (predicted_labels_batch == labels).sum().item()
            num_examples += labels.size(0)

        end_time = time.time()  # End timing after processing
        total_inference_time += end_time - start_time  # Accumulate total inference time

    accuracy = (correct_pred / num_examples) * 100 if num_examples > 0 else 0
    print(f'Accuracy with {attack.__class__.__name__} and Reverse Sigmoid defense: {accuracy:.2f}%')

    # Generate classification report
    target_names = [f"Class {i}" for i in range(10)]
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

    # Calculate and display inference time per 1000 images
    if num_examples > 0:
        time_per_thousand = (total_inference_time / num_examples) * 1000
        print(f"Inference time for 1000 images: {time_per_thousand:.2f} s")

# Usage example with FGSM and PGD
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# Setup the attacks

model.to(device)  # Ensure the model is on the correct device



# Evaluate using FGSM
evaluate_model_with_reverse_sigmoid_and_attack(model, test_loader, device, fgsm_attack)

# Evaluate using PGD
evaluate_model_with_reverse_sigmoid_and_attack(model, test_loader, device, pgd_attack)
