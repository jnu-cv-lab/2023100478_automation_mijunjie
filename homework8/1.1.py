import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')  # 不弹窗
import matplotlib.pyplot as plt

# ====================== 1. 环境检查（自动用GPU） ======================
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ====================== 2. 加载 MNIST 数据集 ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 保存样本图
def plot_samples(dataset, n=8):
    fig, axs = plt.subplots(1, n, figsize=(12, 2))
    for i in range(n):
        img, label = dataset[i]
        axs[i].imshow(img.squeeze(), cmap='gray')
        axs[i].set_title(f'Label: {label}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig("samples.png")  # 自动保存
    plt.close()

plot_samples(full_train)

# ====================== 3. 定义 CNN 模型 ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
print("\n模型结构：")
print(model)

# ====================== 4. 损失函数 & 优化器 ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

# ====================== 5. 训练 ======================
def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def evaluate(loader, model, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

print("\n开始训练...")
for epoch in range(epochs):
    t_loss, t_acc = train_one_epoch(train_loader, model, optimizer, criterion)
    v_loss, v_acc = evaluate(val_loader, model, criterion)

    history['train_loss'].append(t_loss)
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss)
    history['val_acc'].append(v_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | "
          f"Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f}")

# ====================== 测试集结果 ======================
test_loss, test_acc = evaluate(test_loader, model, criterion)
print(f"\n==== 测试集结果 ====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# 保存预测图
def plot_test_pred(loader, model, n=8):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n].to(device), labels[:n].to(device)
    
    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, 1)

    fig, axs = plt.subplots(1, n, figsize=(14, 2))
    for i in range(n):
        img = imgs[i].squeeze().cpu().numpy()
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f'True: {labels[i]}\nPred: {preds[i]}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.close()

plot_test_pred(test_loader, model)

# 保存训练曲线
def plot_curves(history):
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("curves.png")
    plt.close()

plot_curves(history)

print("""
✅ 所有图片已自动保存！
1. samples.png    数据集样本
2. predictions.png 测试集预测结果
3. curves.png     训练曲线
实验完成！
""")