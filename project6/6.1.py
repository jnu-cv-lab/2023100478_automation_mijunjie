import cv2
import numpy as np

# 1. 读取两张图片
img1 = cv2.imread('/home/alexander/cv-course/project6/box_in_scene.png', 0)          # 模板图像 box.png
img2 = cv2.imread('/home/alexander/cv-course/project6/box.png', 0)  # 场景图像 box_in_scene.png

# 2. 创建ORB检测器，设置nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点 + 计算描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 可视化关键点（绘制在彩色图像上）
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

# 5. 输出关键点数量
print("===== ORB 特征检测结果 =====")
print(f"box.png 中的关键点数量：{len(kp1)}")
print(f"box_in_scene.png 中的关键点数量：{len(kp2)}")

# 6. 输出描述子维度
print(f"\nbox.png 描述子形状：{des1.shape}，维度：{des1.shape[1]}")
print(f"box_in_scene.png 描述子形状：{des2.shape}，维度：{des2.shape[1]}")

# 保存可视化图片（提交用）
cv2.imwrite('box_keypoints.png', img1_kp)
cv2.imwrite('box_in_scene_keypoints.png', img2_kp)

# 显示结果
cv2.imshow('box - ORB Keypoints', img1_kp)
cv2.imshow('box_in_scene - ORB Keypoints', img2_kp)

cv2.waitKey(0)
cv2.destroyAllWindows()