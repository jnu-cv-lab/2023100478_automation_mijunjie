import cv2
import numpy as np
import matplotlib.pyplot as plt


# 任务1：使用OpenCV读取一张测试图片 
img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
# 检查图片是否读取成功
if img is None:
    print("错误：图片读取失败，请检查文件路径是否正确！")
    exit()

# 任务2：输出图像基本信息 
print("图像基本信息")
# 获取图像尺寸
height, width, channels = img.shape
# 获取图像数据类型
dtype = img.dtype

print(f"图像宽度：{width} 像素")
print(f"图像高度：{height} 像素")
print(f"图像通道数：{channels}")
print(f"图像数据类型：{dtype}")
print(f"完整尺寸(高, 宽, 通道)：{img.shape}")

# 任务3：显示原图
# 方法1：使用OpenCV显示（推荐）
cv2.imshow('Original Image(OpenCV)', img)
cv2.waitKey(0)  # 等待按键按下再关闭窗口
cv2.destroyAllWindows()  # 释放窗口

# 任务4：转换为灰度图并显示
# 彩色图转灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图
cv2.imshow('Gray Image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 打印灰度图信息
print("\n======== 灰度图信息 ========")
print(f"灰度图尺寸(高, 宽)：{gray_img.shape}")
print(f"灰度图通道数：1 (单通道)")

# 任务5：保存处理结果
# 保存灰度图为新文件
cv2.imwrite('gray_result.jpg', gray_img)
print("\n灰度图已保存为：gray_result.jpg")

# 任务6：NumPy简单操作
print("\nNumPy图像操作")
# 获取图像中心像素值
center_y, center_x = height // 2, width // 2
pixel_value = gray_img[center_y, center_x]
print(f"图像中心({center_x}, {center_y})的灰度像素值：{pixel_value}")

# 操作2：裁剪图像左上角区域（从(0,0)到(150,150)的区域）
# 格式：图像[起始行:结束行, 起始列:结束列]
crop_img = gray_img[0:150, 0:150]

# 显示裁剪后的图像
cv2.imshow('Cropped Image(Top Left 150x150)', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存裁剪后的图片
cv2.imwrite('cropped_result.jpg', crop_img)
print("左上角裁剪图像已保存为：cropped_result.jpg")