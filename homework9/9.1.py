import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# ===================== 设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===================== 复用上次CNN =====================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

# ===================== 数据加载 =====================
def get_mnist():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = datasets.MNIST('./data', train=True, transform=trans, download=True)
    test = datasets.MNIST('./data', train=False, transform=trans, download=True)
    train, val = random_split(full, [54000, 6000])
    return (
        DataLoader(train, 64, shuffle=True),
        DataLoader(val, 64),
        DataLoader(test, 64)
    )

train_loader, val_loader, test_loader = get_mnist()

# ===================== 训练/测试函数 =====================
def train_one_epoch(model, loader, opt, crit):
    model.train()
    loss_sum, acc_sum, cnt = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        acc_sum += (out.argmax(1) == y).sum().item()
        cnt += x.size(0)
    return loss_sum / cnt, acc_sum / cnt

def evaluate(model, loader, crit):
    model.eval()
    loss_sum, acc_sum, cnt = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item() * x.size(0)
            acc_sum += (out.argmax(1) == y).sum().item()
            cnt += x.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return loss_sum / cnt, acc_sum / cnt, np.array(all_preds), np.array(all_labels)

# ===================== 任务2：优化器对比 =====================
def run_optimizer_compare():
    print("\n===== 任务2：优化器对比 =====")
    opts = [
        ("SGD", optim.SGD, {"lr":0.01}),
        ("SGD+Momentum", optim.SGD, {"lr":0.01, "momentum":0.9}),
        ("Adam", optim.Adam, {"lr":0.001})
    ]
    epochs = 5
    crit = nn.CrossEntropyLoss()
    results = {}

    for name, opt_cls, kwargs in opts:
        model = CNN().to(device)
        opt = opt_cls(model.parameters(), **kwargs)
        hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
        print(f"\n--- {name} ---")
        for e in range(epochs):
            tl, ta = train_one_epoch(model, train_loader, opt, crit)
            vl, va, _, _ = evaluate(model, val_loader, crit)
            hist["train_loss"].append(tl)
            hist["val_loss"].append(vl)
            hist["train_acc"].append(ta)
            hist["val_acc"].append(va)
            print(f"Epoch{e+1:2d} | TL:{tl:.4f} VL:{vl:.4f} TA:{ta:.4f} VA:{va:.4f}")
        test_loss, test_acc, _, _ = evaluate(model, test_loader, crit)
        results[name] = (hist, test_loss, test_acc, model)
        print(f"Test Acc: {test_acc:.4f}")
    return results

# ===================== 任务3：学习率对比（Adam） =====================
def run_lr_compare():
    print("\n===== 任务3：学习率对比 =====")
    lrs = [0.1, 0.01, 0.001]
    epochs = 5
    crit = nn.CrossEntropyLoss()
    lr_results = {}

    for lr in lrs:
        model = CNN().to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
        print(f"\n--- lr={lr} ---")
        for e in range(epochs):
            tl, ta = train_one_epoch(model, train_loader, opt, crit)
            vl, va, _, _ = evaluate(model, val_loader, crit)
            hist["train_loss"].append(tl)
            hist["val_loss"].append(vl)
            hist["train_acc"].append(ta)
            hist["val_acc"].append(va)
            print(f"Epoch{e+1:2d} | TL:{tl:.4f} VL:{vl:.4f} TA:{ta:.4f} VA:{va:.4f}")
        lr_results[lr] = hist
    return lr_results

# ===================== 画学习率对比曲线（你要的曲线图） =====================
def plot_lr_compare_curves(lr_results, save_path="lr_compare_curves.png"):
    lrs = list(lr_results.keys())
    colors = ['r', 'g', 'b']
    labels = [f"lr={lr}" for lr in lrs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    for i, lr in enumerate(lrs):
        hist = lr_results[lr]
        ax1.plot(hist["train_loss"], linestyle="-", color=colors[i], label=f"Train {labels[i]}")
        ax1.plot(hist["val_loss"], linestyle="--", color=colors[i], label=f"Val {labels[i]}")
    ax1.set_title("Loss vs Epochs (Adam)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    for i, lr in enumerate(lrs):
        hist = lr_results[lr]
        ax2.plot(hist["train_acc"], linestyle="-", color=colors[i], label=f"Train {labels[i]}")
        ax2.plot(hist["val_acc"], linestyle="--", color=colors[i], label=f"Val {labels[i]}")
    ax2.set_title("Accuracy vs Epochs (Adam)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("已保存学习率对比曲线:", save_path)

# ===================== 任务4：卷积核可视化 =====================
def plot_conv1_kernels(model, save_path="conv1_kernels.png"):
    w = model.features[0].weight.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(w[i,0], cmap="gray")
        axes[i].set_xticks([]); axes[i].set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("已保存卷积核:", save_path)

# ===================== 任务5：Feature map可视化 =====================
def plot_feature_maps(model, loader, save_path="feature_maps.png"):
    x, _ = next(iter(loader))
    img = x[0:1].to(device)
    feat1 = model.features[0](img)
    fm = feat1.detach().cpu().numpy()[0]
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for i in range(8):
        axes[i].imshow(fm[i], cmap="gray")
        axes[i].set_xticks([]); axes[i].set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("已保存特征图:", save_path)

# ===================== 任务6：错误样本可视化 =====================
def plot_error_samples(model, loader, save_path="error_samples.png"):
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            for i in range(len(y)):
                if pred[i] != y[i]:
                    errors.append((x[i].cpu(), y[i].item(), pred[i].item()))
                    if len(errors) >= 8:
                        break
            if len(errors) >= 8:
                break
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for idx, (img, true, pred) in enumerate(errors):
        ax = axes[idx//4, idx%4]
        ax.imshow(img[0], cmap="gray")
        ax.set_title(f"T:{true} P:{pred}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("已保存错误样本:", save_path)

# ===================== 任务7：混淆矩阵 =====================
def plot_confusion_matrix(model, loader, save_path="confusion_matrix.png"):
    _, _, preds, labels = evaluate(model, loader, nn.CrossEntropyLoss())
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("已保存混淆矩阵:", save_path)

# ===================== 主运行 =====================
if __name__ == "__main__":
    # 任务2
    opt_results = run_optimizer_compare()
    best_model = opt_results["Adam"][-1]

    # 任务3
    lr_results = run_lr_compare()
    plot_lr_compare_curves(lr_results)  # ✅ 生成你要的学习率曲线

    # 任务4-7
    plot_conv1_kernels(best_model)
    plot_feature_maps(best_model, test_loader)
    plot_error_samples(best_model, test_loader)
    plot_confusion_matrix(best_model, test_loader)

    print("\n===== 全部任务完成 =====")