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

def integrated_gradients(model, X, feature_names, area_model=None, n_steps=50,
                         output_dir="factor_importance"):
    """Calculate integrated gradients, including area feature importance"""
    os.makedirs(output_dir, exist_ok=True)

    # For gradient calculation, set requires_grad=True
    X_tensor = torch.FloatTensor(X).to(device).requires_grad_(True)

    # Get area features
    if area_model:
        with torch.no_grad():
            _, area_features_orig, _ = area_model(X_tensor)

        # Create feature name list including original features and area feature
        all_feature_names = feature_names.copy()
        all_feature_names.append("Area_Feature")  # English name

        # Create feature name list for display
        display_feature_names = feature_names.copy()
        display_feature_names.append("Landslide Area Feature")  # English name for display
    else:
        all_feature_names = feature_names.copy()
        display_feature_names = feature_names.copy()

    # Define baseline input (all zero features)
    baseline = torch.zeros_like(X_tensor).to(device)

    if area_model:
        with torch.no_grad():
            _, baseline_area_features, _ = area_model(baseline)

    # Integrated gradients for different samples
    all_integrated_grads = []

    # If there's area feature, also store area feature gradients
    if area_model:
        all_area_integrated_grads = []

    # Randomly select 100 samples for calculation (calculating all samples may be time-consuming)
    n_samples = min(100, X.shape[0])
    indices = np.random.choice(X.shape[0], n_samples, replace=False)

    for idx in tqdm(indices, desc="Calculating Integrated Gradients"):
        sample = X_tensor[idx:idx + 1]  # Keep dimensions
        baseline_sample = baseline[idx:idx + 1]

        # Create interpolation points
        alphas = torch.linspace(0, 1, n_steps).to(device)
        interpolated = torch.zeros((n_steps,) + sample.shape).to(device)

        for i, alpha in enumerate(alphas):
            interpolated[i] = baseline_sample + alpha * (sample - baseline_sample)

        interpolated.requires_grad_(True)

        # Calculate predictions
        if area_model:  # Classification model
            # Use same interpolation method to get area features
            with torch.no_grad():
                _, area_features, _ = area_model(interpolated.view(-1, X.shape[1]))
                area_features = area_features.view(n_steps, -1)

            # Calculate area feature gradients
            area_features.requires_grad_(True)
            predictions, _, _ = model(interpolated.view(-1, X.shape[1]), area_features)

            predictions = predictions.view(n_steps, 1)

            # Average gradients across different interpolation points
            if predictions.size(1) == 1:  # Single output (regression or binary classification)
                # Calculate original feature gradients
                gradients = torch.autograd.grad(torch.sum(predictions), interpolated, retain_graph=True)[0]
                # Calculate area feature gradients
                area_gradients = torch.autograd.grad(torch.sum(predictions), area_features)[0]
            else:  # Multi-class classification
                # Calculate original feature gradients
                gradients = torch.autograd.grad(torch.sum(predictions[:, 1]), interpolated, retain_graph=True)[0]
                # Calculate area feature gradients
                area_gradients = torch.autograd.grad(torch.sum(predictions[:, 1]), area_features)[0]

            # Calculate integrated gradients for area feature
            sample_area_feature = area_features_orig[idx:idx + 1]
            baseline_area_feature = baseline_area_features[idx:idx + 1]
            avg_area_gradients = torch.mean(area_gradients, dim=0)
            area_integrated_gradients = (sample_area_feature - baseline_area_feature) * avg_area_gradients
            all_area_integrated_grads.append(area_integrated_gradients.detach().cpu().numpy())

        else:  # Area model
            predictions, _, _ = model(interpolated.view(-1, X.shape[1]))
            predictions = predictions.view(n_steps, 1)

            # Average gradients across different interpolation points
            if predictions.size(1) == 1:  # Single output (regression or binary classification)
                gradients = torch.autograd.grad(torch.sum(predictions), interpolated)[0]
            else:  # Multi-class classification
                gradients = torch.autograd.grad(torch.sum(predictions[:, 1]), interpolated)[0]  # Gradient for positive class

        # Calculate Riemann sum
        avg_gradients = torch.mean(gradients, dim=0)
        integrated_gradients = (sample - baseline_sample) * avg_gradients
        all_integrated_grads.append(integrated_gradients.detach().cpu().numpy())

    # Calculate average integrated gradients across all samples
    all_integrated_grads = np.vstack(all_integrated_grads)
    avg_integrated_grads = np.mean(np.abs(all_integrated_grads), axis=0)  # Take absolute value then average

    # If there's area feature, calculate average integrated gradients for area feature
    if area_model:
        all_area_integrated_grads = np.vstack(all_area_integrated_grads)
        # Average multi-dimensional area features to get overall importance
        avg_area_integrated_grad = np.mean(np.abs(all_area_integrated_grads))

        # Create importance array including original features and area feature
        all_importances = np.append(avg_integrated_grads.flatten(), avg_area_integrated_grad)
    else:
        all_importances = avg_integrated_grads.flatten()

    # Create DataFrame to save results
    importance_df = pd.DataFrame({
        'Feature': display_feature_names,  # Use display names
        'Importance': all_importances
    })

    # Sort by importance in descending order
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Save results
    importance_df.to_csv(os.path.join(output_dir,
                                      "integrated_gradients_area.csv" if not area_model else "integrated_gradients_cls.csv"),
                         index=False)

    # Plot integrated gradients
    plt.figure(figsize=(12, 8))

    # Use different colors to distinguish area feature and original features
    colors = ['#ff7f0e' if feat == 'Landslide Area Feature' else '#1f77b4' for feat in importance_df['Feature']]

    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

    # Add legend
    if area_model:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Original Features'),
            Patch(facecolor='#ff7f0e', label='Landslide Area Feature')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

    plt.title('Integrated Gradients Analysis' + (' - Landslide Susceptibility Model' if area_model else ' - Landslide Area Model'))
    plt.xlabel('Feature Importance (|Gradient|×Input)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             "integrated_gradients_area.png" if not area_model else "integrated_gradients_cls.png"))
    plt.savefig(os.path.join(output_dir,
                             "integrated_gradients_area.pdf" if not area_model else "integrated_gradients_cls.pdf"), dpi=600)
    plt.close()

    return importance_df
