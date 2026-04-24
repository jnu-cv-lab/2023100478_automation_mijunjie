import cv2
import numpy as np

# ===================== 路径 =====================
path1 = "/home/alexander/cv-course/project6/box.png"
path2 = "/home/alexander/cv-course/project6/box_in_scene.png"

# 读取图像
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

# ===================== ORB 特征检测 + 匹配（任务1+2） =====================
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# ===================== 任务3：RANSAC 剔除错误匹配 =====================

# 1. 提取匹配点对坐标（必须步骤）
pts1 = []
pts2 = []
for m in matches:
    pts1.append(kp1[m.queryIdx].pt)  # 图1的点
    pts2.append(kp2[m.trainIdx].pt)  # 图2的点

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# 2. 计算单应矩阵 Homography + RANSAC
H, mask = cv2.findHomography(
    pts1, pts2,
    method=cv2.RANSAC,  # 必须用 RANSAC
    ransacReprojThreshold=5.0  # 重投影误差阈值 5.0
)

# 3. 获取内点mask
matches_mask = mask.ravel().tolist()

# 4. 统计数量
num_matches = len(matches)          # 总匹配数
num_inliers = sum(matches_mask)     # 内点数量
inlier_ratio = num_inliers / num_matches  # 内点比例

# ===================== 输出结果 =====================
print("===== 任务3 RANSAC 剔除误匹配结果 =====")
print(f"总匹配数量：{num_matches}")
print(f"RANSAC 内点数量：{num_inliers}")
print(f"内点比例：{inlier_ratio:.4f}")
print("\nHomography 矩阵 H：")
print(np.round(H, 4))  # 保留4位小数，方便提交

# ===================== 绘制 RANSAC 后的内点匹配图 =====================
img_ransac = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches,
    None,
    matchesMask=matches_mask,  # 只画内点
    flags=cv2.DrawMatchesFlags_DEFAULT
)

# 保存图片（提交用）
cv2.imwrite('/home/alexander/cv-course/project6/ransac_inliers_matches.png', img_ransac)

# 显示
cv2.imshow('RANSAC 内点匹配结果', img_ransac)
cv2.waitKey(0)
cv2.destroyAllWindows()