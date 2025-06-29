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

        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(self.total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2个权重，一个给原始特征，一个给面积特征
            nn.Softmax(dim=1)
        )

        # 特征转换网络
        self.transform_original = nn.Linear(original_dim, original_dim)
        self.transform_area = nn.Linear(area_feature_dim, area_feature_dim)

        # 特征融合后的投影
        self.projection = nn.Linear(self.total_dim, 128)

    def forward(self, original_features, area_features):
        batch_size = original_features.shape[0]

        # 转换特征
        original_transformed = self.transform_original(original_features)
        area_transformed = self.transform_area(area_features)

        # 拼接特征用于计算注意力权重
        combined = torch.cat([original_transformed, area_transformed], dim=1)

        # 计算注意力权重
        attention_weights = self.attention_net(combined)  # [batch_size, 2]

        # 分别为原始特征和面积特征应用权重
        original_attended = original_transformed * attention_weights[:, 0].unsqueeze(1)
        area_attended = area_transformed * attention_weights[:, 1].unsqueeze(1)

        # 拼接加权后的特征
        fused_features = torch.cat([original_attended, area_attended], dim=1)

        # 投影到更高维的空间
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

        # 只对训练集进行数据增强
        if self.is_train and np.random.random() < 0.2:
            # 添加微小的随机噪声
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

        # 只对训练集进行数据增强
        if self.is_train and np.random.random() < 0.2:
            # 添加微小的随机噪声
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
        out += residual  # 残差连接
        out = self.relu(out)
        return out


class AreaResNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_blocks=3):
        super(AreaResNet, self).__init__()

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),  # 添加dropout
            nn.ReLU()
        )

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        # 添加自注意力层
        self.self_attention = SelfAttention(hidden_size)

        # 添加一个额外的dropout
        self.dropout = nn.Dropout(0.3)

        # 输出层
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for i, block in enumerate(self.res_blocks):
            x = block(x)

        # 应用自注意力
        x, attention_weights = self.self_attention(x)

        # 应用dropout (即使在eval模式也可以通过特殊方式激活)
        x = self.dropout(x)

        # 保存特征用于后续模型
        features = x

        output = self.output_layer(x)
        return output, features, attention_weights


class LandslideClassifier(nn.Module):
    def __init__(self, input_size, area_feature_dim=64, hidden_size=128, num_blocks=3, dropout=0.3):
        super(LandslideClassifier, self).__init__()

        # 特征融合注意力机制
        self.fusion_attention = FeatureFusionAttention(input_size, area_feature_dim)

        # 输入层
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(128),  # 融合后的特征维度
            nn.Dropout(dropout)
        )

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(num_blocks)
        ])

        # 自注意力
        self.self_attention = SelfAttention(128)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, area_features):
        # 使用注意力机制融合特征
        fused_features, fusion_weights = self.fusion_attention(x, area_features)

        x = self.input_layer(fused_features)

        for block in self.res_blocks:
            x = block(x)

        # 应用自注意力
        x, self_attention_weights = self.self_attention(x)

        return self.output_layer(x), fusion_weights, self_attention_weights


def train_model(X_train, X_val, y_train, y_val, batch_size=64, num_epochs=200, output_dir="model_output", fold=None):
    # 创建数据集和数据加载器
    train_dataset = AreaDataset(X_train, y_train, is_train=True)
    val_dataset = AreaDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = AreaResNet(X_train.shape[1], hidden_size=128, num_blocks=3).to(device)

    # 损失函数和优化器
    criterion = nn.HuberLoss(delta=1.0)  # 使用Huber损失，对异常值更鲁棒
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # 训练参数
    best_r2 = float('-inf')
    early_stop_patience = 40
    no_improve = 0

    # 记录训练历史
    history = {
        'train_loss': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_rmse': [], 'val_r2': [],
        'lr': []
    }
    model_name_suffix = f"_fold{fold}" if fold is not None else ""
    best_model_path = os.path.join(output_dir, f'best_area_model{model_name_suffix}.pth')
    attention_weights_history = []

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        epoch_attention_weights = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs, _, attention_weights = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 收集注意力权重
            epoch_attention_weights.append(attention_weights.detach().cpu().numpy())

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 累计损失和预测
            train_loss += loss.item()
            all_train_preds.extend(outputs.detach().cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

        # 更新学习率
        scheduler.step()

        # 保存本轮的平均注意力权重
        if len(epoch_attention_weights) > 0:
            avg_attention = np.mean(np.concatenate(epoch_attention_weights, axis=0), axis=0)
            attention_weights_history.append(avg_attention)

        # 计算训练指标
        train_loss /= len(train_loader)
        train_rmse = np.sqrt(mean_squared_error(all_train_labels, all_train_preds))
        train_r2 = r2_score(all_train_labels, all_train_preds)

        # 评估模型
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # 前向传播
                outputs, _, _ = model(batch_x)
                loss = criterion(outputs, batch_y)

                # 累计损失和预测
                val_loss += loss.item()
                all_val_preds.extend(outputs.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())

        # 计算验证指标
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(all_val_labels, all_val_preds))
        val_r2 = r2_score(all_val_labels, all_val_preds)

        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 早停机制
        if val_r2 > best_r2:
            best_r2 = val_r2
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)

            # 保存预测结果
            if fold is not None:
                np.savez(os.path.join(output_dir, f'val_predictions_fold{fold}.npz'),
                         val_preds=all_val_preds,
                         val_labels=all_val_labels)
        else:
            no_improve += 1

        # 打印训练信息
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}')
        print(f'Val Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.2e} | Early Stop: {no_improve}/{early_stop_patience}\n')

        # 早停检查
        if no_improve >= early_stop_patience:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    # 验证集上的最终性能
    model.eval()
    final_val_preds = []
    final_val_features = []
    final_val_attentions = []

    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val).to(device)
        batch_size = 256
        for i in range(0, len(X_val), batch_size):
            batch_X = val_tensor[i:i + batch_size]
            batch_preds, batch_features, batch_attention = model(batch_X)
            final_val_preds.append(batch_preds.cpu().numpy())
            final_val_features.append(batch_features.cpu().numpy())
            final_val_attentions.append(batch_attention.cpu().numpy())

    final_val_preds = np.vstack(final_val_preds)
    final_val_features = np.vstack(final_val_features)
    final_val_attentions = np.vstack(final_val_attentions)

    # 计算所有评估指标
    final_val_r2 = r2_score(y_val, final_val_preds)
    final_val_rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))
    final_val_mse = mean_squared_error(y_val, final_val_preds)
    final_val_mae = mean_absolute_error(y_val, final_val_preds)
    # final_val_nse = nash_sutcliffe_efficiency(y_val, final_val_preds)
    final_val_r = np.corrcoef(y_val.flatten(), final_val_preds.flatten())[0, 1]

    print(f"Fold {fold}: 最终验证集评估结果:")
    print(f"R²: {final_val_r2:.4f}, RMSE: {final_val_rmse:.4f}")
    print(f"MSE: {final_val_mse:.4f}, MAE: {final_val_mae:.4f}")
    # print(f"NSE: {final_val_nse:.4f}, R: {final_val_r:.4f}")

    # 仅为单个折保存训练历史图
    if fold == 0 or fold is None:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history['train_rmse'], label='Train RMSE')
        plt.plot(history['val_rmse'], label='Val RMSE')
        plt.title('RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(history['train_r2'], label='Train R²')
        plt.plot(history['val_r2'], label='Val R²')
        plt.title('R² Score')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_history{model_name_suffix}.png'))
        plt.close()
    if fold == 0 or fold is None:
        plt.figure(figsize=(10, 6))
        attention_weights_history = np.array(attention_weights_history)
        for i in range(attention_weights_history.shape[1]):
            plt.plot(attention_weights_history[:, i], label=f'Dimension {i + 1}')
        plt.title('Attention Weights Evolution During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Average Attention Weight')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'attention_weights_evolution{model_name_suffix}.png'))
        plt.close()

    # 返回所有评估指标
    metrics = {
        'r2': final_val_r2,
        'rmse': final_val_rmse,
        'mse': final_val_mse,
        'mae': final_val_mae,
        # 'nse': final_val_nse,
        'r': final_val_r
    }

    return model, final_val_preds, y_val, metrics, history, final_val_features


def predict_area(model, X):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        predictions = []
        features = []
        attentions = []
        # 分批处理以避免内存问题
        batch_size = 256
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i + batch_size]
            batch_preds, batch_features, batch_attention = model(batch_X)
            predictions.append(batch_preds.cpu().numpy())
            features.append(batch_features.cpu().numpy())
            attentions.append(batch_attention.cpu().numpy())
    return np.vstack(predictions), np.vstack(features), np.vstack(attentions)


def train_classification_model(X_train, X_val, y_train, y_val, area_features_train, area_features_val,
                               batch_size=64, num_epochs=200, output_dir="model_output", fold=None):
    # 创建自定义数据集
    class AttentionDataset(Dataset):
        def __init__(self, X, area_features, y, is_train=False):
            self.X = torch.FloatTensor(X)
            self.area_features = torch.FloatTensor(area_features)
            self.y = torch.FloatTensor(y).unsqueeze(1)
            self.is_train = is_train

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = self.X[idx]
            area_feat = self.area_features[idx]
            y = self.y[idx]

            # 只对训练集进行数据增强
            if self.is_train and np.random.random() < 0.2:
                # 添加微小的随机噪声
                noise_level = 0.03
                x = x + torch.randn_like(x) * noise_level
                area_feat = area_feat + torch.randn_like(area_feat) * noise_level

            return x, area_feat, y

    # 创建数据集和数据加载器
    train_dataset = AttentionDataset(X_train, area_features_train, y_train, is_train=True)
    val_dataset = AttentionDataset(X_val, area_features_val, y_val, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = LandslideClassifier(X_train.shape[1], area_feature_dim=area_features_train.shape[1],
                                hidden_size=256, num_blocks=5, dropout=0.5).to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=10, verbose=True
    )

    # 训练参数
    best_auc = float('-inf')
    early_stop_patience = 30
    no_improve = 0

    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'lr': []
    }

    # 记录注意力权重
    fusion_attention_history = []

    model_name_suffix = f"_fold{fold}" if fold is not None else ""
    best_model_path = os.path.join(output_dir, f'best_cls_model{model_name_suffix}.pth')

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        epoch_fusion_weights = []

        for batch_x, batch_area, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_area = batch_area.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs, fusion_weights, _ = model(batch_x, batch_area)
            loss = criterion(outputs, batch_y)

            # 收集融合注意力权重
            epoch_fusion_weights.append(fusion_weights.detach().cpu().numpy())

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 累计损失和预测
            train_loss += loss.item()
            all_train_preds.extend(outputs.detach().cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

        # 保存本轮的平均融合注意力权重
        if len(epoch_fusion_weights) > 0:
            avg_fusion_weights = np.mean(np.concatenate(epoch_fusion_weights, axis=0), axis=0)
            fusion_attention_history.append(avg_fusion_weights)

        # 计算训练指标
        train_loss /= len(train_loader)
        train_preds_binary = (np.array(all_train_preds) >= 0.5).astype(int)
        train_acc = accuracy_score(all_train_labels, train_preds_binary)
        train_auc = roc_auc_score(all_train_labels, all_train_preds)

        # 评估模型
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_x, batch_area, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_area = batch_area.to(device)
                batch_y = batch_y.to(device)

                # 前向传播
                outputs, _, _ = model(batch_x, batch_area)
                loss = criterion(outputs, batch_y)

                # 累计损失和预测
                val_loss += loss.item()
                all_val_preds.extend(outputs.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())

        # 计算验证指标
        val_loss /= len(val_loader)
        val_preds_binary = (np.array(all_val_preds) >= 0.5).astype(int)
        val_acc = accuracy_score(all_val_labels, val_preds_binary)
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        # 更新学习率
        scheduler.step(val_auc)

        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 早停机制
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)

            # 保存预测结果
            if fold is not None:
                np.savez(os.path.join(output_dir, f'cls_val_predictions_fold{fold}.npz'),
                         val_preds=all_val_preds,
                         val_labels=all_val_labels)
        else:
            no_improve += 1

        # 打印训练信息
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.2e} | Early Stop: {no_improve}/{early_stop_patience}\n')

        # 早停检查
        if no_improve >= early_stop_patience:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    # 验证集上的最终性能
    model.eval()
    final_val_preds = []

    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val).to(device)
        area_tensor = torch.FloatTensor(area_features_val).to(device)
        batch_size = 256
        for i in range(0, len(X_val), batch_size):
            batch_X = val_tensor[i:i + batch_size]
            batch_area = area_tensor[i:i + batch_size]
            # batch_preds = model(batch_X, batch_area)
            outputs, _, _ = model(batch_X, batch_area)
            final_val_preds.append(outputs.cpu().numpy())

    final_val_preds = np.vstack(final_val_preds)
    final_val_preds_binary = (final_val_preds >= 0.5).astype(int)

    # 计算所有评估指标
    final_val_auc = roc_auc_score(y_val, final_val_preds)
    final_val_acc = accuracy_score(y_val, final_val_preds_binary)
    final_val_recall = recall_score(y_val, final_val_preds_binary)
    final_val_precision = precision_score(y_val, final_val_preds_binary)
    final_val_f1 = f1_score(y_val, final_val_preds_binary)
    final_val_mcc = matthews_corrcoef(y_val, final_val_preds_binary)

    print(f"Fold {fold}: 最终验证集评估结果:")
    print(f"AUC: {final_val_auc:.4f}, Accuracy: {final_val_acc:.4f}")
    print(f"Recall: {final_val_recall:.4f}, Precision: {final_val_precision:.4f}")
    print(f"F1: {final_val_f1:.4f}, MCC: {final_val_mcc:.4f}")

    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_val, final_val_preds)

    # 保存ROC曲线数据
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': final_val_auc
    }

    # 仅为单个折保存训练历史图
    if fold == 0 or fold is None:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(history['train_auc'], label='Train AUC')
        plt.plot(history['val_auc'], label='Val AUC')
        plt.title('AUC Score')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cls_training_history{model_name_suffix}.png'))
        plt.close()
    if fold == 0 or fold is None:
        plt.figure(figsize=(10, 6))
        fusion_attention_history = np.array(fusion_attention_history)
        plt.plot(fusion_attention_history[:, 0], label='Original Features')
        plt.plot(fusion_attention_history[:, 1], label='Area Features')
        plt.title('Feature Fusion Attention Weights During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Average Attention Weight')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'fusion_attention_evolution{model_name_suffix}.png'))
        plt.close()

    # 返回所有评估指标
    metrics = {
        'auc': final_val_auc,
        'accuracy': final_val_acc,
        'recall': final_val_recall,
        'precision': final_val_precision,
        'f1': final_val_f1,
        'mcc': final_val_mcc
    }

    return model, final_val_preds, y_val, metrics, history, roc_data


