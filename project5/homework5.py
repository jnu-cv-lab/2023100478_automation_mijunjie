import cv2
import numpy as np

# ====================== 1. 生成测试图 ======================
h, w = 600, 600
img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白底

# 矩形
cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)

# 圆
cv2.circle(img, (450, 200), 80, (255, 0, 0), 2)

# 平行线（水平）
cv2.line(img, (100, 400), (300, 400), (0, 128, 0), 2)
cv2.line(img, (100, 450), (300, 450), (0, 128, 0), 2)

# 垂直线（垂直相交）
cv2.line(img, (400, 400), (400, 550), (128, 0, 128), 2)
cv2.line(img, (350, 475), (450, 475), (128, 0, 128), 2)

# 定义用于变换的源点（矩形的四个角）
src_points = np.float32([[100, 100], [300, 100], [300, 300], [100, 300]])

# --- 相似变换 ---
# 使用 getRotationMatrix2D 创建旋转+缩放的变换矩阵
center = (w // 2, h // 2)  # 旋转中心
angle = 15  # 旋转角度
scale = 1.2  # 缩放因子
M_sim = cv2.getRotationMatrix2D(center, angle, scale)
# 为了兼容仿射变换函数，我们需要扩展M_sim
M_sim_full = np.vstack([M_sim, [0, 0, 1]]) # 添加齐次坐标的一行
img_sim = cv2.warpAffine(img, M_sim, (w, h)) # warpAffine 接受 2x3 矩阵
cv2.putText(img_sim, 'Similarity (Rot+Scale)', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# --- 仿射变换 ---
# 为了清晰地展示剪切，只移动一个角点
# src[:3] -> dst_aff
src_aff = src_points[:3] # 取三个点
dst_aff = np.float32([[120, 100], [320, 100], [300, 300]]) # 左上角向右移动，右上角也向右移动更多
M_aff = cv2.getAffineTransform(src_aff, dst_aff)
img_aff = cv2.warpAffine(img, M_aff, (w, h))
cv2.putText(img_aff, 'Affine (Shear)', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# --- 透视变换 ---
dst_per = np.float32([[100, 120], [300, 90], [280, 320], [120, 290]])
M_per = cv2.getPerspectiveTransform(src_points, dst_per)
img_per = cv2.warpPerspective(img, M_per, (w, h))
cv2.putText(img_per, 'Perspective', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# --- 在原图上也添加标签 ---
cv2.putText(img, 'Original', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)


# ====================== 拼接显示 ======================
top_row = np.hstack([img, img_sim])
bottom_row = np.hstack([img_aff, img_per])
result_image = np.vstack([top_row, bottom_row])

cv2.imshow('Transformation Comparison: Original | Similarity | Affine | Perspective', result_image)
cv2.imwrite('corrected_transform_compare.png', result_image)
print("Transformed image saved as 'corrected_transform_compare.png'")
cv2.waitKey(0)
cv2.destroyAllWindows()