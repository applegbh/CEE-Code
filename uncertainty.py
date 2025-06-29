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

def predict_with_uncertainty(area_model, cls_model, X, n_samples=100, dropout_prob=0.2):
    """
    Calculate prediction confidence intervals using Monte Carlo dropout

    Parameters:
    area_model: Trained area prediction model
    cls_model: Trained classification model
    X: Input features (already standardized)
    n_samples: Number of Monte Carlo samples
    dropout_prob: Dropout probability

    Returns:
    area_predictions: Multiple samples of area prediction results, shape=(n_samples, n_samples)
    lsm_predictions: Multiple samples of susceptibility prediction results, shape=(n_samples, n_samples)
    """
    # Set models to evaluation mode but enable dropout
    area_model.eval()
    cls_model.eval()

    # Enable dropout
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    enable_dropout(area_model)
    enable_dropout(cls_model)

    # Prepare arrays to store multiple sampling results
    n_samples_per_batch = 10  # Samples per batch to reduce memory usage
    n_batches = n_samples // n_samples_per_batch

    X_tensor = torch.FloatTensor(X).to(device)
    n_instances = X.shape[0]

    area_predictions = np.zeros((n_instances, n_samples))
    lsm_predictions = np.zeros((n_instances, n_samples))

    # Perform Monte Carlo sampling in batches
    with torch.no_grad():
        for batch in tqdm(range(n_batches), desc="Monte Carlo Sampling"):
            for i in range(n_samples_per_batch):
                sample_idx = batch * n_samples_per_batch + i

                # Process in batches to avoid memory issues
                batch_size = 256
                batch_area_preds = []
                batch_area_features = []
                batch_lsm_preds = []

                for j in range(0, n_instances, batch_size):
                    batch_X = X_tensor[j:j + batch_size]

                    # 1. Perform area prediction
                    outputs, features, _ = area_model(batch_X)
                    batch_area_preds.append(outputs.cpu().numpy())
                    batch_area_features.append(features.cpu().numpy())

                # Combine batch results
                area_preds = np.vstack(batch_area_preds).flatten()
                area_features = np.vstack(batch_area_features)

                # Process in batches again for classification prediction
                for j in range(0, n_instances, batch_size):
                    batch_X = X_tensor[j:j + batch_size]
                    batch_area_feat = torch.FloatTensor(area_features[j:j + batch_size]).to(device)

                    # 2. Perform susceptibility prediction
                    outputs, _, _ = cls_model(batch_X, batch_area_feat)
                    batch_lsm_preds.append(outputs.cpu().numpy())

                # Combine batch results
                lsm_preds = np.vstack(batch_lsm_preds).flatten()

                # Store results
                area_predictions[:, sample_idx] = area_preds
                lsm_predictions[:, sample_idx] = lsm_preds

    return area_predictions, lsm_predictions


def calculate_confidence_intervals(predictions):
    """
    Calculate 5%, 50%, and 95% confidence intervals

    Parameters:
    predictions: Prediction array with shape (n_instances, n_samples)

    Returns:
    lower: 5% confidence interval
    median: 50% confidence interval (median)
    upper: 95% confidence interval
    """
    lower = np.percentile(predictions, 5, axis=1)
    median = np.percentile(predictions, 50, axis=1)
    upper = np.percentile(predictions, 95, axis=1)

    return lower, median, upper


# Generate confidence maps and save results
def generate_confidence_maps(original_gdf, area_predictions, lsm_predictions, output_dir="confidence_maps"):
    """
    Generate confidence maps and save results

    Parameters:
    original_gdf: Original GeoDataFrame
    area_predictions: Area prediction array with shape (n_instances, n_samples)
    lsm_predictions: Susceptibility prediction array with shape (n_instances, n_samples)
    output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate confidence intervals
    area_lower, area_median, area_upper = calculate_confidence_intervals(area_predictions)
    lsm_lower, lsm_median, lsm_upper = calculate_confidence_intervals(lsm_predictions)

    # Convert back to original scale (only for area)
    area_lower_orig = np.expm1(area_lower)
    area_median_orig = np.expm1(area_median)
    area_upper_orig = np.expm1(area_upper)

    # Calculate uncertainty
    area_uncertainty = area_upper - area_lower
    area_relative_uncertainty = area_uncertainty / area_median
    lsm_uncertainty = lsm_upper - lsm_lower

    # Add to GeoDataFrame
    result_gdf = original_gdf.copy()

    # Add area confidence intervals
    result_gdf['Area_5pct'] = area_lower
    result_gdf['Area_50pct'] = area_median
    result_gdf['Area_95pct'] = area_upper
    result_gdf['Area_Uncert'] = area_uncertainty
    result_gdf['Area_RelUnc'] = area_relative_uncertainty

    # Add original scale area confidence intervals
    result_gdf['OrgArea_5pct'] = area_lower_orig
    result_gdf['OrgArea_50pct'] = area_median_orig
    result_gdf['OrgArea_95pct'] = area_upper_orig

    # Add susceptibility confidence intervals
    result_gdf['LSM_5pct'] = lsm_lower
    result_gdf['LSM_50pct'] = lsm_median
    result_gdf['LSM_95pct'] = lsm_upper
    result_gdf['LSM_Uncert'] = lsm_uncertainty

    # Save results
    result_gdf.to_file(os.path.join(output_dir, "confidence_intervals.shp"))

    # Create visualizations
    create_confidence_visualizations(result_gdf, output_dir)

    return result_gdf


def create_confidence_visualizations(gdf, output_dir):
    """Create visualizations of confidence intervals"""
    # Area prediction visualization
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # 5% confidence interval (area)
    gdf.plot(column='Area_5pct', cmap='viridis', legend=True, ax=axs[0, 0])
    axs[0, 0].set_title('Landslide Area 5% Confidence Interval (Log Scale)')

    # 50% confidence interval (area)
    gdf.plot(column='Area_50pct', cmap='viridis', legend=True, ax=axs[0, 1])
    axs[0, 1].set_title('Landslide Area 50% Confidence Interval (Log Scale)')

    # 95% confidence interval (area)
    gdf.plot(column='Area_95pct', cmap='viridis', legend=True, ax=axs[1, 0])
    axs[1, 0].set_title('Landslide Area 95% Confidence Interval (Log Scale)')

    # Uncertainty (area)
    gdf.plot(column='Area_Uncert', cmap='viridis', legend=True, ax=axs[1, 1])
    axs[1, 1].set_title('Landslide Area Uncertainty (90% Confidence Interval Width)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "area_confidence_maps.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, "area_confidence_maps.pdf"), dpi=600)
    plt.close()

    # Susceptibility prediction visualization
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # 5% confidence interval (susceptibility)
    gdf.plot(column='LSM_5pct', cmap='viridis', legend=True, ax=axs[0, 0])
    axs[0, 0].set_title('Landslide Susceptibility 5% Confidence Interval')

    # 50% confidence interval (susceptibility)
    gdf.plot(column='LSM_50pct', cmap='viridis', legend=True, ax=axs[0, 1])
    axs[0, 1].set_title('Landslide Susceptibility 50% Confidence Interval')

    # 95% confidence interval (susceptibility)
    gdf.plot(column='LSM_95pct', cmap='viridis', legend=True, ax=axs[1, 0])
    axs[1, 0].set_title('Landslide Susceptibility 95% Confidence Interval')

    # Uncertainty (susceptibility)
    gdf.plot(column='LSM_Uncert', cmap='viridis', legend=True, ax=axs[1, 1])
    axs[1, 1].set_title('Landslide Susceptibility Uncertainty (90% Confidence Interval Width)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lsm_confidence_maps.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "lsm_confidence_maps.pdf"), dpi=600)
    plt.close()