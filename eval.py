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

# Set Torch Home
os.environ['TORCH_HOME'] = '/Users/parth/mnt/shared'

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.alexnet(pretrained=True)
model.eval()
model.to(device)

# Load dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),  # Adjust based on your model input size
    transforms.ToTensor(),
])

dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
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
        cpu_usage_before = psutil.cpu_percent(interval=None)
        memory_usage_before = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Store predictions and labels
        all_preds.append(preds.cpu().numpy())  # Append predictions for this batch
        all_labels.append(labels.cpu().numpy())  # Append labels for this batch

        # Track CPU and Memory usage after inference
        cpu_usage_after = psutil.cpu_percent(interval=None)
        memory_usage_after = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB

        # Store metrics for the current batch
        metrics.append({
            "Batch": len(metrics) + 1,
            "CPU Usage Before (%)": cpu_usage_before,
            "CPU Usage After (%)": cpu_usage_after,
            "Memory Usage Before (MB)": memory_usage_before,
            "Memory Usage After (MB)": memory_usage_after
        })

        # Print relevant metrics for the current batch
        print(f"Batch {len(metrics)}:")
        print(f"  CPU Usage Before: {cpu_usage_before:.2f}%")
        print(f"  CPU Usage After: {cpu_usage_after:.2f}%")
        print(f"  Memory Usage Before: {memory_usage_before:.2f} MB")
        print(f"  Memory Usage After: {memory_usage_after:.2f} MB")

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

# AUC-ROC calculation
try:
    # This part may need adjustment as MNIST is a multi-class problem.
    # AUC-ROC is typically calculated for binary classification.
    # You can skip it or compute it for each class separately.
    auc_roc = roc_auc_score(all_labels, torch.softmax(outputs, dim=1).cpu().numpy(), multi_class='ovr')
except Exception as e:
    print(f"AUC-ROC calculation failed: {e}")
    auc_roc = None

# Save metrics to CSV
df_metrics = pd.DataFrame(metrics)
df_metrics['Total Inference Time (s)'] = inference_time
df_metrics['Accuracy (%)'] = accuracy
df_metrics['Precision (%)'] = precision
df_metrics['Recall (%)'] = recall
df_metrics['F1 Score'] = f1
df_metrics['AUC-ROC'] = auc_roc

# Print all the metrics
print("Total Inference Time (s):", inference_time)
print("Accuracy (%):", accuracy)
print("Precision (%):", precision)
print("Recall (%):", recall)
print("F1 Score:", f1)
print("AUC-ROC:", auc_roc)
print("Confusion Matrix:\n", conf_matrix)

# Specify the output CSV file name
df_metrics.to_csv('performance_metrics.csv', index=False)

print("Performance metrics saved to performance_metrics.csv")
