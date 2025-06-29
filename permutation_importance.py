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
def permutation_importance(model, X, y, feature_names, area_model=None, n_repeats=10,
                           batch_size=256, output_dir="factor_importance"):
    """Calculate permutation importance, including area feature importance"""
    os.makedirs(output_dir, exist_ok=True)

    X_tensor = torch.FloatTensor(X).to(device)

    # First get area features
    if area_model:  # For classification model, get area features first
        with torch.no_grad():
            _, area_features, _ = area_model(X_tensor)
            area_features = area_features.cpu().numpy()

    # Create feature name list including original features and area feature
    all_feature_names = feature_names.copy()
    if area_model:
        all_feature_names.append("Area_Feature")  # English name

    # Create feature name list for display
    display_feature_names = feature_names.copy()
    if area_model:
        display_feature_names.append("Landslide Area Feature")  # English name for display

    results = np.zeros((len(all_feature_names), n_repeats))

    # First get baseline predictions
    model.eval()
    with torch.no_grad():
        if area_model:  # For classification model, get area features first
            area_tensor = torch.FloatTensor(area_features).to(device)
            baseline_preds, _, _ = model(X_tensor, area_tensor)
        else:  # Area model predicts directly
            baseline_preds, _, _ = model(X_tensor)

        baseline_preds = baseline_preds.cpu().numpy().flatten()

    # Calculate baseline performance metric
    if area_model:  # Classification model uses AUC
        baseline_score = roc_auc_score(y, baseline_preds)
    else:  # Area model uses R²
        baseline_score = r2_score(y, baseline_preds)

    # Calculate permutation importance for each feature
    for i, feature_name in enumerate(tqdm(all_feature_names, desc="Calculating Permutation Importance")):
        for r in range(n_repeats):
            if feature_name == "Area_Feature":
                # Permute area feature
                # 1. First permute original features to generate new area features
                X_permuted = X.copy()
                perm_idx = np.random.permutation(len(X))
                X_permuted_tensor = torch.FloatTensor(X_permuted[perm_idx]).to(device)

                with torch.no_grad():
                    _, permuted_area_features, _ = area_model(X_permuted_tensor)
                    permuted_area_features = permuted_area_features.cpu().numpy()

                # 2. Use original features and permuted area features for prediction
                with torch.no_grad():
                    permuted_area_tensor = torch.FloatTensor(permuted_area_features).to(device)
                    permuted_preds, _, _ = model(X_tensor, permuted_area_tensor)
                    permuted_preds = permuted_preds.cpu().numpy().flatten()
            else:
                # Create permuted feature data
                X_permuted = X.copy()
                orig_feature_idx = all_feature_names.index(feature_name)
                X_permuted[:, orig_feature_idx] = np.random.permutation(X_permuted[:, orig_feature_idx])
                X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)

                # Get predictions using permuted features
                with torch.no_grad():
                    if area_model:  # Classification model
                        # First get features from area model
                        _, permuted_area_features, _ = area_model(X_permuted_tensor)
                        permuted_area_features = permuted_area_features.cpu().numpy()
                        permuted_area_tensor = torch.FloatTensor(permuted_area_features).to(device)

                        # Then make classification prediction
                        permuted_preds, _, _ = model(X_permuted_tensor, permuted_area_tensor)
                    else:  # Area model
                        permuted_preds, _, _ = model(X_permuted_tensor)

                    permuted_preds = permuted_preds.cpu().numpy().flatten()

            # Calculate permuted performance metric
            if area_model:
                permuted_score = roc_auc_score(y, permuted_preds)
            else:
                permuted_score = r2_score(y, permuted_preds)

            # Calculate importance (baseline score minus permuted score)
            results[i, r] = baseline_score - permuted_score

    # Calculate mean and standard deviation of importance for each feature
    importance_mean = results.mean(axis=1)
    importance_std = results.std(axis=1)

    # Create DataFrame to save results
    importance_df = pd.DataFrame({
        'Feature': display_feature_names,  # Use display names
        'Importance': importance_mean,
        'Std': importance_std
    })

    # Sort by importance in descending order
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Save results
    importance_df.to_csv(os.path.join(output_dir,
                                      "permutation_importance_area.csv" if not area_model else "permutation_importance_cls.csv"),
                         index=False)

    # Plot permutation importance
    plt.figure(figsize=(8, 6))

    # Use different colors to distinguish area feature and original features
    colors = ['#ff7f0e' if feat == 'Landslide Area Feature' else '#1f77b4' for feat in importance_df['Feature']]

    plt.barh(importance_df['Feature'], importance_df['Importance'],
             xerr=importance_df['Std'], capsize=5, color=colors)

    # Add legend
    if area_model:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Original Features'),
            Patch(facecolor='#ff7f0e', label='Landslide Area Feature')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

    plt.title('Permutation Importance Analysis' + (' - Landslide Susceptibility Model' if area_model else ' - Landslide Area Model'))
    plt.xlabel('Importance (Performance Decrease)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             "permutation_importance_area.png" if not area_model else "permutation_importance_cls.png"))
    plt.savefig(os.path.join(output_dir,
                             "permutation_importance_area.pdf" if not area_model else "permutation_importance_cls.pdf"), dpi=600)
    plt.close()

    return importance_df