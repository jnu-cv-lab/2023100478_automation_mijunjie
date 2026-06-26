import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 1. 实验参数配置（报告必填） =====================
# 棋盘格内角点 9列×6行
CHESSBOARD_W = 9
CHESSBOARD_H = 6
# 方格实际尺寸 mm
SQUARE_SIZE = 22
# 图片文件夹
IMG_PATH = "./imgs"
# 输出文件夹
OUT_PATH = "./output"
os.makedirs(OUT_PATH, exist_ok=True)

# 构建棋盘格三维世界坐标（标定板坐标系Z=0）
objp = np.zeros((CHESSBOARD_W * CHESSBOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_W, 0:CHESSBOARD_H].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# 存储所有图片的3D点、2D角点
obj_points = []
img_points = []
img_list = []

# ===================== 2. 遍历图片+检测角点+亚像素优化 =====================
# 亚像素迭代终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 读取所有图片
for fname in os.listdir(IMG_PATH):
    if fname.endswith((".jpg", ".png", ".jpeg")):
        full_path = os.path.join(IMG_PATH, fname)
        img = cv2.imread(full_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list.append((fname, img, gray))

        # 检测棋盘内角点
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_W, CHESSBOARD_H))
        if ret:
            obj_points.append(objp)
            # 亚像素精度优化角点
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners_sub)

            # 绘制并保存角点图（报告需要2张以上角点检测图）
            draw_img = img.copy()
            cv2.drawChessboardCorners(draw_img, (CHESSBOARD_W, CHESSBOARD_H), corners_sub, ret)
            cv2.imwrite(os.path.join(OUT_PATH, f"corner_{fname}"), draw_img)
            print(f"成功检测角点：{fname}")
        else:
            print(f"角点检测失败：{fname}")

if len(obj_points) == 0:
    raise Exception("无有效标定图片，请重新拍摄棋盘格")

# ===================== 3. 相机标定，求解内参、畸变、外参、重投影误差 =====================
gray_shape = img_list[0][2].shape[::-1]
# 标定核心函数
ret_error, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray_shape, None, None
)

# 打印标定核心结果（报告直接复制）
print("="*50)
print("相机内参矩阵 K：")
print(K)
print("\n畸变系数 D [k1,k2,p1,p2,k3]：")
print(D.ravel())
print(f"\n平均重投影误差：{ret_error:.4f} 像素")
print("="*50)

# 保存参数到txt，方便写报告
with open(os.path.join(OUT_PATH, "calib_result.txt"), "w", encoding="utf-8") as f:
    f.write(f"棋盘格内角点：{CHESSBOARD_W}×{CHESSBOARD_H}\n")
    f.write(f"方格边长：{SQUARE_SIZE} mm\n")
    f.write(f"图像分辨率：{gray_shape[0]} × {gray_shape[1]}\n")
    f.write("相机内参矩阵 K:\n")
    f.write(str(K) + "\n")
    f.write("畸变系数 D [k1,k2,p1,p2,k3]:\n")
    f.write(str(D.ravel()) + "\n")
    f.write(f"平均重投影误差：{ret_error:.4f} px\n")

# ===================== 4. 图像去畸变 undistort，原图与校正图对比 =====================
# 取第一张有效图片做去畸变演示
demo_img = img_list[0][1]
h, w = demo_img.shape[:2]
# 优化相机矩阵（裁剪黑边）
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
# 去畸变
undist_img = cv2.undistort(demo_img, K, D, None, new_K)
# 裁剪无效黑边
x, y, w_valid, h_valid = roi
undist_img_crop = undist_img[y:y+h_valid, x:x+w_valid]

# 保存原图、去畸变图
cv2.imwrite(os.path.join(OUT_PATH, "origin_demo.jpg"), demo_img)
cv2.imwrite(os.path.join(OUT_PATH, "undist_demo.jpg"), undist_img_crop)

# matplotlib绘制对比图（可直接放进实验报告）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("原始图像")
plt.imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("去畸变校正图像")
plt.imshow(cv2.cvtColor(undist_img_crop, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.savefig(os.path.join(OUT_PATH, "compare_undist.png"), dpi=150, bbox_inches="tight")
plt.show()