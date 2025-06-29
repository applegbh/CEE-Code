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


def spatial_feature_importance(model, X, feature_names, gdf, area_model=None,
                               output_dir="factor_importance"):
    """Generate spatial distribution maps of feature importance, including area feature importance"""
    os.makedirs(output_dir, exist_ok=True)

    # Get area features
    X_tensor = torch.FloatTensor(X).to(device)

    if area_model:
        with torch.no_grad():
            _, area_features, _ = area_model(X_tensor)
            area_features = area_features.cpu().numpy()

        # Create feature name list including original features and area feature
        all_feature_names = feature_names.copy()
        all_feature_names.append("Area_Feature")  # English name

        # Create feature name list for display (keep English names for chart display)
        display_feature_names = feature_names.copy()
        display_feature_names.append("Landslide Area Feature")
    else:
        all_feature_names = feature_names.copy()
        display_feature_names = feature_names.copy()
        area_features = None

    feature_importances = []
    model.eval()

    batch_size = 256
    for i in range(0, X.shape[0], batch_size):
        batch_X = X_tensor[i:i + batch_size]
        batch_X.requires_grad_(True)

        if area_model:  # Classification model
            # Get area features
            batch_area = torch.FloatTensor(area_features[i:i + batch_size]).to(device)
            batch_area.requires_grad_(True)
            predictions, _, _ = model(batch_X, batch_area)
        else:  # Area model
            predictions, _, _ = model(batch_X)

        # Calculate gradients
        gradients = []
        for j in range(min(batch_size, len(batch_X))):
            if area_model:  # Classification model (binary)
                # Calculate original feature gradients
                grad_x = torch.autograd.grad(predictions[j], batch_X, retain_graph=True)[0][j].abs().cpu().numpy()
                # Calculate area feature gradients
                grad_area = torch.autograd.grad(predictions[j], batch_area, retain_graph=True)[0][j].abs().cpu().numpy()
                # Average area feature gradients to get overall feature importance
                grad_area_mean = np.mean(grad_area)
                # Combine original feature gradients and area feature gradients
                grad = np.append(grad_x, grad_area_mean)
            else:  # Area model (regression)
                grad = torch.autograd.grad(predictions[j], batch_X, retain_graph=True)[0][j].abs().cpu().numpy()

            gradients.append(grad)

        feature_importances.extend(gradients)

    # Convert feature importance to DataFrame
    importance_df = pd.DataFrame(feature_importances, columns=all_feature_names)

    # Add feature importance to GeoDataFrame
    for feature in all_feature_names:
        gdf[f'Imp_{feature}'] = importance_df[feature].values

    # Save feature importance GeoDataFrame
    importance_gdf = gdf.copy()
    importance_gdf.to_file(os.path.join(output_dir, "spatial_feature_importance.shp"))

    # Create separate maps for each feature
    for i, feature in enumerate(all_feature_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        importance_gdf.plot(column=f'Imp_{feature}', cmap='viridis',
                            legend=True, ax=ax)

        # Use display name (can include English) as title
        display_name = display_feature_names[i] if i < len(display_feature_names) else feature
        plt.title(f'Spatial Distribution of Feature Importance - {display_name}')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"spatial_importance_{feature}.png"))
        plt.savefig(os.path.join(output_dir, f"spatial_importance_{feature}.pdf"),dpi=600)
        plt.close()

    # Map original feature names back to names used for other analyses
    if area_model:
        # Create new DataFrame with display names for other analysis functions
        disp_importance_df = pd.DataFrame(feature_importances, columns=display_feature_names)
        return disp_importance_df
    else:
        return importance_df