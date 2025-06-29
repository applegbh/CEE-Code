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

rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei',
                                   'WenQuanYi Micro Hei', 'sans-serif']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


import matplotlib.pyplot as plt
from tqdm import tqdm
def pdp_feature_importance(model, X, feature_names, area_model=None,
                           output_dir="factor_importance", n_points=50):
    """Calculate Partial Dependence Plots (PDP) and save results as CSV and images"""
    os.makedirs(output_dir, exist_ok=True)

    # Create PyTorch model wrapper for sklearn
    class TorchModelWrapper:
        def __init__(self, torch_model, area_model=None):
            self.torch_model = torch_model
            self.area_model = area_model

        # Add fit method to satisfy sklearn interface requirements
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            X_tensor = torch.FloatTensor(X).to(device)
            self.torch_model.eval()

            with torch.no_grad():
                if self.area_model:
                    # For classification model, get area features first
                    _, area_features, _ = self.area_model(X_tensor)
                    area_features = area_features.cpu().numpy()
                    area_tensor = torch.FloatTensor(area_features).to(device)

                    # Then make classification prediction
                    predictions, _, _ = self.torch_model(X_tensor, area_tensor)
                else:
                    # Area model predicts directly
                    predictions, _, _ = self.torch_model(X_tensor)

                return predictions.cpu().numpy().flatten()

    # Create model wrapper
    model_wrapper = TorchModelWrapper(model, area_model)

    # Manually calculate PDP instead of using sklearn's partial_dependence
    all_pdp_values = {}

    for feature_idx, feature_name in enumerate(tqdm(feature_names, desc="Calculating PDP Feature Importance")):
        # Create feature value grid
        feature_values = np.linspace(
            np.min(X[:, feature_idx]),
            np.max(X[:, feature_idx]),
            n_points
        )

        # Calculate average prediction for each feature value
        pdp_values = []
        for value in feature_values:
            # Create modified dataset where target feature is set to current value
            X_modified = X.copy()
            X_modified[:, feature_idx] = value

            # Get predictions
            predictions = model_wrapper.predict(X_modified)

            # Calculate average prediction
            avg_prediction = np.mean(predictions)
            pdp_values.append(avg_prediction)

        # Save to dictionary
        all_pdp_values[feature_name] = {
            'feature_values': feature_values,
            'pdp_values': np.array(pdp_values)
        }

        # Plot PDP
        plt.figure(figsize=(10, 6))
        plt.plot(feature_values, pdp_values)
        plt.title(f'PDP - {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Prediction Change')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save images
        model_type = "area" if area_model is None else "cls"
        plt.savefig(os.path.join(output_dir, f"pdp_{model_type}_{feature_name}.png"))
        plt.savefig(os.path.join(output_dir, f"pdp_{model_type}_{feature_name}.pdf"), dpi=600)
        plt.close()

        # Save CSV data
        pd.DataFrame({
            'feature_value': feature_values,
            'pdp_value': pdp_values
        }).to_csv(os.path.join(output_dir, f"pdp_{model_type}_{feature_name}.csv"), index=False)

    # For classification model, also calculate PDP for area feature
    if area_model:
        # Get area features
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            _, area_features, _ = area_model(X_tensor)
            area_features = area_features.cpu().numpy()

        # Calculate PDP for area feature
        # Create area feature value grid
        area_feature_values = np.linspace(
            np.min(area_features),
            np.max(area_features),
            n_points
        )

        area_pdp_values = []
        for value in area_feature_values:
            # Create modified area features
            modified_area_features = np.ones_like(area_features) * value

            # Get predictions
            predictions = []
            batch_size = 256
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_X = torch.FloatTensor(X[i:end_idx]).to(device)
                batch_area = torch.FloatTensor(modified_area_features[i:end_idx]).to(device)

                with torch.no_grad():
                    batch_preds, _, _ = model(batch_X, batch_area)
                    predictions.append(batch_preds.cpu().numpy())

            all_preds = np.vstack(predictions).flatten()
            avg_prediction = np.mean(all_preds)
            area_pdp_values.append(avg_prediction)

        # Plot PDP
        plt.figure(figsize=(10, 6))
        plt.plot(area_feature_values, area_pdp_values)
        plt.title('Partial Dependence Plot - Landslide Area Feature')
        plt.xlabel('Landslide Area Feature')
        plt.ylabel('Prediction Change')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save images
        plt.savefig(os.path.join(output_dir, "pdp_cls_Area_Feature.png"))
        plt.savefig(os.path.join(output_dir, "pdp_cls_Area_Feature.pdf"), dpi=600)
        plt.close()

        # Save CSV data
        pd.DataFrame({
            'feature_value': area_feature_values,
            'pdp_value': area_pdp_values
        }).to_csv(os.path.join(output_dir, "pdp_cls_Area_Feature.csv"), index=False)

        # Add area feature to results dictionary
        all_pdp_values['Area_Feature'] = {
            'feature_values': area_feature_values,
            'pdp_values': np.array(area_pdp_values)
        }

    # Create summary CSV file
    summary_data = {}

    # Add feature value columns
    for feature_name in all_pdp_values:
        feature_values = all_pdp_values[feature_name]['feature_values']
        summary_data[f'{feature_name}_value'] = pd.Series(feature_values)

    # Add PDP value columns
    for feature_name in all_pdp_values:
        pdp_values = all_pdp_values[feature_name]['pdp_values']
        summary_data[f'{feature_name}_pdp'] = pd.Series(pdp_values)

    # Create DataFrame and save
    model_type = "area" if area_model is None else "cls"
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f"pdp_all_features_{model_type}.csv"), index=False)

    return all_pdp_values
