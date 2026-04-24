import cv2
import numpy as np

# ===================== 路径 =====================
path1 = "/home/alexander/cv-course/project6/box.png"
path2 = "/home/alexander/cv-course/project6/box_in_scene.png"

# 读取图像（灰度图用于特征检测，彩色图用于绘制）
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
img2_color = cv2.imread(path2)  # 彩色图，用于绘制框

# ===================== ORB 检测 + 匹配 + RANSAC 求单应矩阵 H =====================
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# 计算单应矩阵 H
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# ===================== 任务4：目标定位 =====================

# 1. 获取 box.png 的四个角点（左上角、右上角、右下角、左下角）
h, w = img1.shape  # 获取模板图的高和宽
pts_corner = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 2. 使用 Homography 矩阵投影到场景图中
dst_corner = cv2.perspectiveTransform(pts_corner, H)

# 3. 在场景图上绘制出目标的四边形边框
img_result = cv2.polylines(
    img2_color,
    [np.int32(dst_corner)],
    True,          # True = 闭合图形
    (0, 255, 0),   # 绿色
    3,             # 线条粗细
    cv2.LINE_AA
)

# ===================== 保存 & 显示结果 =====================
cv2.imwrite('/home/alexander/cv-course/project6/box_detection_result.png', img_result)
cv2.imshow('目标定位结果', img_result)

print("===== 任务4 目标定位完成 =====")
print("✅ 已在场景图中绘制出 box 目标的边框")
print("✅ 保存图片：box_detection_result.png")

cv2.waitKey(0)
cv2.destroyAllWindows()