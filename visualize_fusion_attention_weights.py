import io
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef, roc_curve)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.inspection import partial_dependence
from datetime import datetime
import joblib
import matplotlib
from matplotlib import rcParams


def visualize_fusion_attention_weights(model, X, area_features, output_dir="factor_importance"):
    """Visualize feature fusion attention weights"""
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        area_tensor = torch.FloatTensor(area_features).to(device)

        # Get fusion attention weights
        _, fusion_weights, _ = model(X_tensor, area_tensor)
        fusion_weights = fusion_weights.cpu().numpy()

    # Plot fusion weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(fusion_weights[:, 0], bins=50, alpha=0.6, label='Original Feature Weights')
    plt.hist(fusion_weights[:, 1], bins=50, alpha=0.6, label='Area Feature Weights')
    plt.title('Feature Fusion Attention Weight Distribution')
    plt.xlabel('Attention Weight')
    plt.ylabel('Sample Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fusion_attention_distribution.png"))
    plt.close()

    # Calculate and print average attention weights
    avg_orig_weight = np.mean(fusion_weights[:, 0])
    avg_area_weight = np.mean(fusion_weights[:, 1])

    print(f"Average attention weight for original features: {avg_orig_weight:.4f}")
    print(f"Average attention weight for area features: {avg_area_weight:.4f}")

    # Create DataFrame to represent attention weight importance
    attention_importance = pd.DataFrame({
        'Feature': ['Original Features', 'Landslide Area Feature'],
        'Attention_Weight': [avg_orig_weight, avg_area_weight]
    })

    # Save results
    attention_importance.to_csv(os.path.join(output_dir, "fusion_attention_importance.csv"), index=False)

    # Plot attention weight bar chart
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e']
    plt.bar(attention_importance['Feature'], attention_importance['Attention_Weight'], color=colors)
    plt.title('Feature Fusion Attention Weight Importance')
    plt.xlabel('Feature Type')
    plt.ylabel('Average Attention Weight')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "fusion_attention_importance.png"))
    plt.close()

    return fusion_weights, attention_importance