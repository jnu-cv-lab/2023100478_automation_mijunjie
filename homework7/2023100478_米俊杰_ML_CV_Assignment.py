# -*- coding: utf-8 -*-
"""
传统机器学习方法在手写数字图像分类中的应用
适配WSL环境，解决中文乱码、空图、tkinter依赖问题
"""

# 1. 优先设置Matplotlib后端（解决WSL空图问题）
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，直接保存图片，避免依赖tkinter
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 改用英文安全字体，避免乱码
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 导入所有分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 任务1：数据准备
# =============================================================================
print("="*60)
print("任务1：数据准备")
print("="*60)

digits = load_digits()
X = digits.data       # 特征向量 (1797, 64)
y = digits.target     # 标签 (1797,)
images = digits.images  # 原始二维图像 (1797, 8, 8)

print(f"数据集中图像总数：{len(images)} 张")
print(f"每张图像的大小：{images.shape[1]} × {images.shape[2]} 像素")
print(f"所有类别标签：{np.unique(y)}")
print(f"类别总数：{len(np.unique(y))} 类")

# 显示12张样本图像
plt.figure(figsize=(12, 4))
for i in range(12):
    plt.subplot(2, 6, i+1)
    plt.imshow(images[i], cmap='gray_r')
    plt.title(f"Label: {y[i]}", fontsize=10)  # 改用英文标题，避免乱码
    plt.axis('off')
plt.suptitle('Handwritten Digit Samples', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("task1_samples.png", dpi=150)  # 保存图片，不弹出窗口
plt.close()  # 关闭figure，避免内存泄漏

# =============================================================================
# 任务2：数据划分
# =============================================================================
print("\n" + "="*60)
print("任务2：数据划分")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"训练集样本数量：{X_train.shape[0]} 个")
print(f"测试集样本数量：{X_test.shape[0]} 个")
print("\n训练集用途：用于训练机器学习模型，让模型学习数据中的模式和参数")
print("测试集用途：用于评估模型的泛化能力，模拟模型在从未见过的新数据上的表现")

# =============================================================================
# 任务3：特征表示
# =============================================================================
print("\n" + "="*60)
print("任务3：特征表示")
print("="*60)

print("1. 8×8图像转换为64维向量的方法：")
print("   将二维图像矩阵按行优先的顺序展开为一维向量。例如：")
print("   图像矩阵 [[a11,a12,...,a18], [a21,a22,...,a28], ..., [a81,a82,...,a88]]")
print("   转换后向量 [a11,a12,...,a18,a21,a22,...,a28,...,a81,a82,...,a88]")
print(f"   本数据集转换后每个样本的特征维度：{X.shape[1]}")

print("\n2. 传统机器学习需要特征转换的原因：")
print("   传统机器学习模型（如SVM、逻辑回归、决策树等）的输入必须是固定长度的一维向量，")
print("   无法直接处理二维图像矩阵结构，因此需要将图像展平为特征向量。")

print("\n3. 原始像素作为特征的优点：")
print("   - 提取过程简单，无需复杂的特征工程")
print("   - 完整保留了图像的所有原始像素信息")
print("   - 计算速度快，适合小规模数据集")

print("\n4. 原始像素作为特征的局限：")
print("   - 对图像的平移、旋转、缩放等空间变换非常敏感")
print("   - 特征维度较高，容易产生维度灾难")
print("   - 只能提取低级像素信息，无法捕捉图像的高级语义特征（如边缘、纹理、形状）")

# =============================================================================
# 任务4：模型训练（6种方法全部实现）
# =============================================================================
print("\n" + "="*60)
print("任务4：模型训练与测试")
print("="*60)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', gamma='scale', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

accuracy_results = {}
predict_results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = test_acc
    predict_results[model_name] = y_pred
    print(f"{model_name:20s} 测试准确率：{test_acc:.4f}")

# =============================================================================
# 任务5：结果比较
# =============================================================================
print("\n" + "="*60)
print("任务5：模型结果比较")
print("="*60)

print("\n不同模型测试准确率对比表：")
print("| 模型 | 测试准确率 |")
print("|------|------------|")
for model_name, acc in accuracy_results.items():
    print(f"| {model_name} | {acc:.4f} |")

best_model_name = max(accuracy_results, key=accuracy_results.get)
worst_model_name = min(accuracy_results, key=accuracy_results.get)
acc_difference = max(accuracy_results.values()) - min(accuracy_results.values())

print(f"\n准确率最高的模型：{best_model_name}，准确率为 {accuracy_results[best_model_name]:.4f}")
print(f"准确率最低的模型：{worst_model_name}，准确率为 {accuracy_results[worst_model_name]:.4f}")
print(f"模型间最大准确率差异：{acc_difference:.4f}，表现差异较为明显")

print("\n模型表现差异的原因分析：")
print("1. SVM和KNN对高维像素特征的拟合能力较强，能有效捕捉不同数字的像素分布模式")
print("2. 朴素贝叶斯假设特征之间相互独立，而图像像素之间存在强空间相关性，导致分类效果较差")
print("3. 单棵决策树容易过拟合训练数据中的噪声，泛化能力较弱")
print("4. 随机森林通过集成多棵决策树的预测结果，有效降低了过拟合风险，表现优于单棵决策树")

# =============================================================================
# 任务6：错误样本分析（解决空图问题）
# =============================================================================
print("\n" + "="*60)
print("任务6：错误样本分析")
print("="*60)

best_y_pred = predict_results[best_model_name]
confusion_mat = confusion_matrix(y_test, best_y_pred)

# 绘制混淆矩阵（解决空图问题）
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=digits.target_names)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'{best_model_name} Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig("task6_confusion_matrix.png", dpi=150)
plt.close()

# 错误样本分析
error_indexes = np.where(best_y_pred != y_test)[0]
print(f"\n错误分类的样本总数：{len(error_indexes)} 个")
print(f"模型错误率：{len(error_indexes)/len(y_test):.4f}")

# 显示前8个错误样本
if len(error_indexes) > 0:
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(error_indexes[:8]):
        plt.subplot(2, 4, i+1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray_r')
        plt.title(f"True: {y_test[idx]}\nPred: {best_y_pred[idx]}", fontsize=12)
        plt.axis('off')
    plt.suptitle(f'{best_model_name} Misclassified Samples', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("task6_error_samples.png", dpi=150)
    plt.close()
else:
    print("模型无错误分类样本！")

print("\n错误样本误判原因分析：")
print("1. 形状相似的数字容易被混淆，例如8和3、9和4、5和6，它们的像素分布较为接近")
print("2. 部分手写数字书写不规范，笔画模糊或变形，导致模型无法正确识别")
print("3. 原始像素特征缺乏空间不变性，数字的轻微偏移或旋转就会导致像素值发生较大变化")

# =============================================================================
# 思考题答案
# =============================================================================
print("\n" + "="*60)
print("思考题答案")
print("="*60)

print("1. 为什么传统机器学习方法需要先把图像转换为特征向量？")
print("答：传统机器学习模型的输入接口要求是固定长度的一维数值向量，无法直接处理二维图像矩阵结构。")
print("将图像展平为特征向量是为了满足模型的输入格式要求，让模型能够对图像数据进行计算和学习。")

print("\n2. KNN、SVM、决策树和随机森林的分类思想有什么不同？")
print("答：- KNN：基于实例的学习，通过计算待分类样本与训练集中所有样本的距离，选择最近的K个样本进行投票分类")
print("    - SVM：寻找能够最大化不同类别间隔的超平面作为分类边界，对高维数据和小样本数据表现良好")
print("    - 决策树：通过递归地选择最优特征进行分裂，将数据划分为不同的子集，直到满足停止条件")
print("    - 随机森林：集成学习方法，通过随机采样样本和特征训练多棵决策树，最终通过投票得到分类结果")

print("\n3. 为什么单棵决策树容易过拟合？")
print("答：单棵决策树在训练过程中会不断分裂节点，直到所有训练样本都被正确分类或满足停止条件。")
print("这会导致模型学习到训练集中的噪声和异常点，而不是数据的普遍规律，因此在测试集上的泛化能力较差。")

print("\n4. 为什么随机森林通常比单棵决策树更稳定？")
print("答：随机森林通过两个随机性来提高模型稳定性：一是对训练集进行有放回的随机采样，二是在每个节点分裂时随机选择特征子集。")
print("这使得每棵决策树都有不同的学习视角，通过集成多棵树的预测结果，可以有效降低单棵树的过拟合风险，提高模型的泛化能力和稳定性。")

print("\n5. 如果图像发生平移、旋转或缩放，直接使用原始像素作为特征会有什么问题？")
print("答：原始像素特征记录的是每个位置的像素值，对图像的空间变换非常敏感。")
print("同一数字经过平移、旋转或缩放后，像素值会发生显著变化，导致模型无法识别出这是同一个数字，分类准确率会大幅下降。")

print("\n" + "="*60)
print("程序运行完成！图片已保存为：")
print("task1_samples.png, task6_confusion_matrix.png, task6_error_samples.png")
print("="*60)