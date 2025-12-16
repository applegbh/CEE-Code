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
rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    def forward(self, x):
        # x shape: [batch_size, hidden_size]
        batch_size = x.shape[0]

        # Reshape for attention calculation
        x_reshaped = x.view(batch_size, 1, -1)

        # Create Q, K, V
        Q = self.query(x_reshaped)  # [batch_size, 1, hidden_size]
        K = self.key(x_reshaped)  # [batch_size, 1, hidden_size]
        V = self.value(x_reshaped)  # [batch_size, 1, hidden_size]

        # Attention scores
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, 1, 1]
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to V
        x = torch.matmul(attention, V).squeeze(1)  # [batch_size, hidden_size]

        return x, attention

class FeatureFusionAttention(nn.Module):
    def __init__(self, original_dim, area_feature_dim):
        super(FeatureFusionAttention, self).__init__()
        self.original_dim = original_dim
        self.area_feature_dim = area_feature_dim
        self.total_dim = original_dim + area_feature_dim

        self.attention_net = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  
            nn.Softmax(dim=1)
        )

        self.transform_original = nn.Linear(original_dim, original_dim)
        self.transform_area = nn.Linear(area_feature_dim, area_feature_dim)
        self.projection = nn.Linear(self.total_dim, 128)

    def forward(self, original_features, area_features):
        batch_size = original_features.shape[0]

        original_transformed = self.transform_original(original_features)
        area_transformed = self.transform_area(area_features)

        combined = torch.cat([original_transformed, area_transformed], dim=1)

        attention_weights = self.attention_net(combined)  # [batch_size, 2]

        original_attended = original_transformed * attention_weights[:, 0].unsqueeze(1)
        area_attended = area_transformed * attention_weights[:, 1].unsqueeze(1)

        fused_features = torch.cat([original_attended, area_attended], dim=1)

        output = self.projection(fused_features)

        return output, attention_weights


class AreaDataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.is_train and np.random.random() < 0.2:
            noise_level = 0.03
            x = x + torch.randn_like(x) * noise_level

        return x, y


class ClassificationDataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.is_train and np.random.random() < 0.2:
            noise_level = 0.03
            x = x + torch.randn_like(x) * noise_level

        return x, y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out += residual 
        out = self.relu(out)
        return out


class AreaResNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_blocks=3):
        super(AreaResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3), 
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        self.self_attention = SelfAttention(hidden_size)

        self.dropout = nn.Dropout(0.3)

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for i, block in enumerate(self.res_blocks):
            x = block(x)

        x, attention_weights = self.self_attention(x)
        x = self.dropout(x)
        features = x

        output = self.output_layer(x)
        return output, features, attention_weights


class LandslideClassifier(nn.Module):
    def __init__(self, input_size, area_feature_dim=64, hidden_size=128, num_blocks=3, dropout=0.3):
        super(LandslideClassifier, self).__init__()

        self.fusion_attention = FeatureFusionAttention(input_size, area_feature_dim)


        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(128), 
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(num_blocks)
        ])

        self.self_attention = SelfAttention(128)
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, area_features):

        fused_features, fusion_weights = self.fusion_attention(x, area_features)

        x = self.input_layer(fused_features)

        for block in self.res_blocks:
            x = block(x)
        x, self_attention_weights = self.self_attention(x)

        return self.output_layer(x), fusion_weights, self_attention_weights




