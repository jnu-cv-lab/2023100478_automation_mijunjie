import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================== 设备设置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch版本:", torch.__version__)
print("使用设备:", device)

# ====================== CNN模型（支持MNIST和CIFAR-10） ======================
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
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
        # 关键修复：用自适应池化固定输出为 4x4，避免手动计算维度
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256), # 现在输入固定是128*4*4=2048，不会变了
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

# ====================== 训练&评估函数 ======================
def train(model, loader, opt, criterion):
    model.train()
    loss_sum, acc_sum, cnt = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        acc_sum += (out.argmax(1) == y).sum().item()
        cnt += x.size(0)
    return loss_sum / cnt, acc_sum / cnt

def test(model, loader, criterion):
    model.eval()
    loss_sum, acc_sum, cnt = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            acc_sum += (out.argmax(1) == y).sum().item()
            cnt += x.size(0)
    return loss_sum / cnt, acc_sum / cnt

# ====================== 数据集加载 ======================
def get_mnist():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('./data', train=True, transform=trans, download=True)
    test = datasets.MNIST('./data', train=False, transform=trans, download=True)
    train, val = random_split(train, [54000, 6000])
    return (DataLoader(train,64,True), DataLoader(val,64), DataLoader(test,64))

def get_cifar10():
    train_trans = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    train = datasets.CIFAR10('./data',train=True,transform=train_trans,download=True)
    test = datasets.CIFAR10('./data',train=False,transform=test_trans,download=True)
    train, val = random_split(train, [45000,5000])
    return (DataLoader(train,64,True), DataLoader(val,64), DataLoader(test,64))

# ====================== 运行实验 ======================
def run(name, model, train_loader, val_loader, test_loader, optimizer, epochs=10):
    print(f"\n===== {name} =====")
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}

    for ep in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = test(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {ep+1:2d} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

    test_loss, test_acc = test(model, test_loader, criterion)
    print(f"【最终测试】{name} 准确率: {test_acc:.4f}")

    plt.figure(figsize=(12,4))
    plt.subplot(121); plt.plot(history['train_loss'],label='train'); plt.plot(history['val_loss'],label='val'); plt.title('Loss')
    plt.subplot(122); plt.plot(history['train_acc'],label='train'); plt.plot(history['val_acc'],label='val'); plt.title('Acc')
    plt.legend(); plt.tight_layout(); plt.savefig(f'{name}.png'); plt.close()
    return test_acc

# ====================== 【运行所有任务】 ======================
if __name__ == '__main__':
    m_train, m_val, m_test = get_mnist()
    c_train, c_val, c_test = get_cifar10()

    # 基础任务
    model_base = CNN(1).to(device)
    opt_base = optim.Adam(model_base.parameters(), lr=0.001)
    acc_mnist = run("MNIST_基础", model_base, m_train, m_val, m_test, opt_base, epochs=5)

    # 进阶2：优化器对比 SGD
    model_sgd = CNN(1).to(device)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
    acc_sgd = run("MNIST_SGD", model_sgd, m_train, m_val, m_test, opt_sgd, epochs=5)

    # 进阶2：优化器对比 Adam
    model_adam = CNN(1).to(device)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    acc_adam = run("MNIST_Adam", model_adam, m_train, m_val, m_test, opt_adam, epochs=5)

    # 进阶3：CIFAR10（现在不会报错了）
    model_cifar = CNN(3).to(device)
    opt_cifar = optim.Adam(model_cifar.parameters(), lr=0.001)
    acc_cifar = run("CIFAR10", model_cifar, c_train, c_val, c_test, opt_cifar, epochs=10)

    # 输出最终结果（直接写报告）
    print("\n" + "="*50)
    print("实验全部完成！")
    print(f"MNIST 基础: {acc_mnist:.2%}")
    print(f"MNIST SGD: {acc_sgd:.2%}")
    print(f"MNIST Adam: {acc_adam:.2%}")
    print(f"CIFAR10: {acc_cifar:.2%}")
    print("="*50)