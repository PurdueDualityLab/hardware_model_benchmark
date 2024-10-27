import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torchvision.transforms as transforms
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import Inception_V3_Weights

from board_utlis import Board, check_RPI_CPU_temp
from dataset import ImageNetValidation

# ------------------------------------------------
# --------- CHange this for every run ------------
# ------------------------------------------------
board_name = Board.RASPBERRY_PI_4

model_name = 'inception_v3'
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

dataset_name = 'imagenet_validation'
num_samples = 1000

TORCH_HOME = os.path.expanduser("~/shared")
read_cpu_temp = check_RPI_CPU_temp

# ------------------------------------------------




os.environ['TORCH_HOME'] = TORCH_HOME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
model.eval()
model.to(device)


# --------------------------------------------------------------
# --------- CHange this if dataset is not image net ------------
# --------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


data_dir = os.path.join(TORCH_HOME, 'ILSVRC2012_img_val')
ground_truth_file = os.path.join(TORCH_HOME, 'ILSVRC2012_validation_ground_truth.txt')
dataset = ImageNetValidation(
    data_dir,
    ground_truth_file,
    transform=transform,
    num_samples=num_samples
)
# --------------------------------------------------------------

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Performance metrics storage
metrics = []
all_preds = []  # To store all predictions
all_labels = []  # To store all labels

# Run inference
start_time = time.time()
with torch.no_grad():
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Track CPU and Memory usage before inference
        batch_start_time = time.time()
        cpu_usage_before = psutil.cpu_percent(interval=None)
        memory_usage_before = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        # Store predictions and labels
        all_preds.append(preds.cpu().numpy())  # Append predictions for this batch
        all_labels.append(labels.cpu().numpy())  # Append labels for this batch

        # Track CPU and Memory usage after inference
        cpu_usage_after = psutil.cpu_percent(interval=None)
        memory_usage_after = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
        batch_end_time = time.time()
        batch_total_time = batch_end_time - batch_start_time
        cpu_temp, _ = read_cpu_temp()

        # Store metrics for the current batch
        metrics.append({
            "Batch": len(metrics) + 1,
            "CPU Usage Before (%)": cpu_usage_before,
            "CPU Usage After (%)": cpu_usage_after,
            "Memory Usage Before (MB)": memory_usage_before,
            "Memory Usage After (MB)": memory_usage_after,
            "CPU Temperature (°C)": cpu_temp,
            "Inference Time (s)": batch_total_time
        })

        # Print relevant metrics for the current batch
        print(f"Batch {len(metrics)}:")
        print(f"  CPU Usage Before: {cpu_usage_before:.2f}%")
        print(f"  CPU Usage After: {cpu_usage_after:.2f}%")
        print(f"  Memory Usage Before: {memory_usage_before:.2f} MB")
        print(f"  Memory Usage After: {memory_usage_after:.2f} MB")
        print(f"  Accuracy: {accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()) * 100:.2f}%")

# Concatenate all predictions and labels after inference
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate total performance metrics
end_time = time.time()
inference_time = end_time - start_time

# Task performance metrics
accuracy = accuracy_score(all_labels, all_preds) * 100
precision = precision_score(all_labels, all_preds, average='weighted') * 100
recall = recall_score(all_labels, all_preds, average='weighted') * 100
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Save metrics to CSV
df_metrics = pd.DataFrame(metrics)
df_metrics['Total Inference Time (s)'] = inference_time
df_metrics['Accuracy (%)'] = accuracy
df_metrics['Precision (%)'] = precision
df_metrics['Recall (%)'] = recall
df_metrics['F1 Score'] = f1

# Print all the metrics
print("Total Inference Time (s):", inference_time)
print("Accuracy (%):", accuracy)
print("Precision (%):", precision)
print("Recall (%):", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# Specify the output CSV file name
output_dir = os.path.join('output', board_name.value)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

formatted_start_time = time.strftime("%b_%d_%H:%M:%S", time.localtime(start_time))
output_file = os.path.join(output_dir, f'{model_name}_{dataset_name}_performance_metrics_{formatted_start_time}.csv')
df_metrics.to_csv(output_file, index=False)
print("Performance metrics saved to CSV:", output_file)
