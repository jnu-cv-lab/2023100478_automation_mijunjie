import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"

# 模型超参（实验文档推荐值）
INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
NUM_CLASSES = 6
DROPOUT = 0.1

# 训练超参
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 20

# 加载标签映射
with open(f"{DATA_DIR}/label_map.json", "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
LABEL_MAP = {int(k): v for k, v in LABEL_MAP.items()}

# ===================== 自定义数据集 =====================
class BadmintonSkeletonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return feat, label

# ===================== Skeleton Transformer 模型 =====================
class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = TARGET_FRAMES
        # 1. 线性嵌入 132 -> d_model
        self.linear_emb = nn.Linear(INPUT_DIM, D_MODEL)
        # 2. 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, self.T, D_MODEL))
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=NUM_LAYERS)
        # 4. 分类头
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        # x: [B, 30, 132]
        B = x.shape[0]
        # 特征嵌入
        x = self.linear_emb(x)  # [B, 30, 128]
        # 加位置编码
        x = x + self.pos_emb
        # Transformer Encoder
        x = self.transformer_encoder(x)  # [B, 30, 128]
        # 全局均值池化
        x = torch.mean(x, dim=1)  # [B, 128]
        # 分类
        x = self.dropout(x)
        logits = self.classifier(x)  # [B, 6]
        return logits

# ===================== 加载数据 =====================
train_dataset = BadmintonSkeletonDataset(f"{DATA_DIR}/X_train.npy", f"{DATA_DIR}/y_train.npy")
test_dataset = BadmintonSkeletonDataset(f"{DATA_DIR}/X_test.npy", f"{DATA_DIR}/y_test.npy")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===================== 模型、损失、优化器 =====================
model = SkeletonTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===================== 训练循环 =====================
def train_one_epoch():
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for feats, labels in train_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        logits = model(feats)
        loss = criterion(logits, labels)

        # 反向传播 + 更新
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += feats.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# ===================== 测试/验证函数 =====================
def evaluate():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)

            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += feats.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, all_labels, all_preds

# ===================== 主训练流程 =====================
if __name__ == "__main__":
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    print(f"使用设备: {DEVICE}")
    print("===== 开始训练 =====")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch()
        test_loss, test_acc, _, _ = evaluate()

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}\n")

    # 保存模型
    torch.save(model.state_dict(), "./badminton_transformer.pth")
    print("模型已保存为 badminton_transformer.pth")

    # 输出混淆矩阵 & 分类报告
    _, _, y_true, y_pred = evaluate()
    print("===== 混淆矩阵 =====")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n===== 分类报告 =====")
    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values())))

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Test Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./train_curve.png")
    plt.show()

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(LABEL_MAP.values()),
                yticklabels=list(LABEL_MAP.values()))
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Pred Label")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png")
    plt.show()