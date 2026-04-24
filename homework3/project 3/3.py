import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# 1. 创建保存结果的文件夹
# ==========================
os.makedirs("result", exist_ok=True)

# ==========================
# 2. 评价指标：MSE、PSNR
# ==========================
def get_mse_psnr(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return 0, 100
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return mse, psnr

# ==========================
# 3. 傅里叶变换频谱（中心化 + 对数）
# ==========================
def fft_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1e-8)
    return magnitude

# ==========================
# 4. DCT 变换 + 低频能量占比
# ==========================
def dct_process(img):
    img_f32 = img.astype(np.float32)
    dct = cv2.dct(img_f32)
    dct_log = np.log(np.abs(dct) + 1e-8)
    return dct, dct_log

def low_freq_ratio(dct, ratio=0.1):
    h, w = dct.shape
    th, tw = int(h * ratio), int(w * ratio)
    total = np.sum(dct ** 2)
    low = np.sum(dct[:th, :tw] ** 2)
    return low / total

# ==========================
# 5. 读取灰度图（请放入 lena.png）
# ==========================
img = cv2.imread("/home/alexander/cv-course/project 3/result/lena.png", cv2.IMREAD_GRAYSCALE)
H, W = img.shape
print(f"原图尺寸：{H}x{W}")

# ==========================
# 6. 下采样 1/2（两种方式）
# ==========================
# 方式1：不做预滤波直接下采样
img_direct = cv2.resize(img, (W//2, H//2), interpolation=cv2.INTER_NEAREST)

# 方式2：高斯平滑后下采样（抗混叠）
img_blur = cv2.GaussianBlur(img, (5,5), 0.8)
img_blur_down = cv2.resize(img_blur, (W//2, H//2), interpolation=cv2.INTER_NEAREST)

# ==========================
# 7. 三种插值恢复（使用高斯平滑后的下采样图）
# ==========================
methods = {
    "最近邻": cv2.INTER_NEAREST,
    "双线性": cv2.INTER_LINEAR,
    "双三次": cv2.INTER_CUBIC
}

restored = {}
for name, m in methods.items():
    restored[name] = cv2.resize(img_blur_down, (W, H), interpolation=m)

# ==========================
# 8. 计算 MSE / PSNR
# ==========================
print("\n===== MSE & PSNR =====")
for name, res in restored.items():
    mse, psnr = get_mse_psnr(img, res)
    print(f"{name}：MSE={mse:.2f}, PSNR={psnr:.2f}dB")

# ==========================
# 9. 傅里叶频谱
# ==========================
fft_ori = fft_spectrum(img)
fft_down = fft_spectrum(img_blur_down)
fft_bilinear = fft_spectrum(restored["双线性"])

# ==========================
# 10. DCT 与能量占比
# ==========================
dct_ori, dct_log_ori = dct_process(img)
dct_bili, dct_log_bili = dct_process(restored["双线性"])

r_ori = low_freq_ratio(dct_ori)
r_bili = low_freq_ratio(dct_bili)

print("\n===== DCT 低频能量占比(10%) =====")
print(f"原图：{r_ori:.4f}")
print(f"双线性恢复：{r_bili:.4f}")

# ==========================
# 11. 绘图并保存到 result/ 文件夹
# ==========================
plt.figure(figsize=(16, 10))

# 第1行：空间域图像
plt.subplot(3,5,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(3,5,2)
plt.imshow(img_blur_down, cmap='gray')
plt.title("Downsampled (Gaussian)")

plt.subplot(3,5,3)
plt.imshow(restored["最近邻"], cmap='gray')
plt.title("Restored - Nearest")

plt.subplot(3,5,4)
plt.imshow(restored["双线性"], cmap='gray')
plt.title("Restored - Bilinear")

plt.subplot(3,5,5)
plt.imshow(restored["双三次"], cmap='gray')
plt.title("Restored - Bicubic")

# 第2行：FFT 频谱
plt.subplot(3,5,6)
plt.imshow(fft_ori, cmap='gray')
plt.title("FFT Original")

plt.subplot(3,5,7)
plt.imshow(fft_down, cmap='gray')
plt.title("FFT Downsampled")

plt.subplot(3,5,8)
plt.imshow(fft_bilinear, cmap='gray')
plt.title("FFT Bilinear")

# 第3行：DCT
plt.subplot(3,5,11)
plt.imshow(dct_log_ori, cmap='gray')
plt.title("DCT Original")

plt.subplot(3,5,12)
plt.imshow(dct_log_bili, cmap='gray')
plt.title("DCT Bilinear")

plt.tight_layout()
plt.savefig("result/all_result.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 全部完成！图片已保存到 result/ 文件夹")
原图尺寸：500x500

===== MSE & PSNR =====
最近邻：MSE=64.12, PSNR=30.06dB
双线性：MSE=54.87, PSNR=30.74dB
双三次：MSE=47.14, PSNR=31.40dB

===== DCT 低频能量占比(10%) =====
原图：0.9906
双线性恢复：0.9952

✅ 全部完成！图片已保存到 result/ 文件夹
