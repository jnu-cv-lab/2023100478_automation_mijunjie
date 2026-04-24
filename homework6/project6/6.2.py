import cv2
import numpy as np

# ===================== 图片路径=====================
path1 = "/home/alexander/cv-course/project6/box.png"
path2 = "/home/alexander/cv-course/project6/box_in_scene.png"

# 读取灰度图
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

# ===================== 任务1：ORB 特征检测 =====================
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 输出关键点数量 & 描述子维度
print("===== 任务1 特征检测结果 =====")
print(f"box.png 关键点数量：{len(kp1)}")
print(f"box_in_scene.png 关键点数量：{len(kp2)}")
print(f"描述子维度：{des1.shape[1]} 维\n")

# 保存任务1可视化图
img1_kp = cv2.drawKeypoints(img1, kp1, None, (0,255,0), 0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, (0,255,0), 0)
cv2.imwrite("/home/alexander/cv-course/project6/box_keypoints.png", img1_kp)
cv2.imwrite("/home/alexander/cv-course/project6/box_in_scene_keypoints.png", img2_kp)

# ===================== 任务2：ORB 特征匹配 =====================
# 1. 创建暴力匹配器（ORB必须用NORM_HAMMING）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 2. 匹配
matches = bf.match(des1, des2)

# 3. 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 4. 输出总匹配数
print("===== 任务2 特征匹配结果 =====")
print(f"总匹配数量：{len(matches)}")

# ===================== ✅ 生成两张匹配图 =====================

# 图1：ORB 初始全部匹配图（所有匹配点）
img_all_matches = cv2.drawMatches(
    img1, kp1, img2, kp2,
    matches,        # 全部匹配
    None, flags=2
)
cv2.imwrite("/home/alexander/cv-course/project6/orb_all_matches.png", img_all_matches)

# 图2：前 50 个最优匹配图
img_top50_matches = cv2.drawMatches(
    img1, kp1, img2, kp2,
    matches[:50],   # 前50个
    None, flags=2
)
cv2.imwrite("/home/alexander/cv-course/project6/orb_top50_matches.png", img_top50_matches)

# ===================== 显示 =====================
cv2.imshow("ORB 全部初始匹配", img_all_matches)
cv2.imshow("ORB 前50最优匹配", img_top50_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()