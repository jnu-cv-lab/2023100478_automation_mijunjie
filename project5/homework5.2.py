import cv2
import numpy as np
import os

# --- 全局变量 ---
selected_points = []
original_image = None
image_path = ""

def draw_points_on_image(img, points):
    """辅助函数：在图像副本上绘制所有已选的点"""
    img_with_points = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img_with_points, (x, y), 10, (0, 255, 0), -1)  # 绿色实心圆
        # 可选：为每个点添加数字标签
        # cv2.putText(img_with_points, str(i+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img_with_points

def mouse_callback(event, x, y, flags, param):
    """鼠标回调，直接在原图上选点"""
    global selected_points, original_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append((x, y))
            # 重新绘制所有点并更新窗口
            img_to_show = draw_points_on_image(original_image, selected_points)
            cv2.imshow('Select 4 Corners of Document', img_to_show)
            print(f"已选择点 {len(selected_points)}: ({x}, {y})")

            if len(selected_points) == 4:
                print("\n--- 开始校正 ---")
                correct_perspective()

def order_points(pts):
    """
    自动将4个点排序为：左上 → 右上 → 右下 → 左下
    此函数通过几何方法计算，更稳定
    """
    # 将点转换为NumPy数组
    rect = np.zeros((4, 2), dtype="float32")

    # 计算点的中心
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    # 计算每个点相对于中心的角度
    angles = np.arctan2(pts[:, 1] - center_y, pts[:, 0] - center_x)
    # 将角度归一化到 0-2π
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    # 根据角度排序
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]

    # 找到排序后点中最接近 -π/4 (315度) 的作为左上，以此类推
    # 更稳健的方式是使用距离中心的象限判断
    # 但为了简化和兼容性强，我们可以基于排序后的角度位置来分配
    # 0度附近 -> 右上, 90度附近 -> 右下, 180度附近 -> 左下, 270度 (-90度)附近 -> 左上
    # 排序后的索引 0,1,2,3 分别对应角度最小的点、第二小...最大的点
    # 我们可以通过判断它们的实际坐标来更精确地分配
    # 一个更简单且常用的方法是：
    # s = x + y (左上角最小), d = x - y (右上角最小)
    s = sorted_pts.sum(axis=1) # x + y
    d = np.diff(sorted_pts, axis=1) # x - y

    # 左上角是 x+y 最小的
    rect[0] = sorted_pts[np.argmin(s)]
    # 右下角是 x+y 最大的
    rect[2] = sorted_pts[np.argmax(s)]
    # 右上角是 x-y 最小的 (因为x>y时差为正，x<y时差为负)
    remaining_indices = [i for i in range(4) if not (np.array_equal(sorted_pts[i], rect[0]) or np.array_equal(sorted_pts[i], rect[2]))]
    remaining_pts = sorted_pts[remaining_indices]
    temp_d = remaining_pts[:, 0] - remaining_pts[:, 1] # Recalculate diff for remaining two
    rect[1] = remaining_pts[np.argmin(temp_d)] # Right-top has smallest x-y
    rect[3] = remaining_pts[np.argmax(temp_d)] # Left-bottom has largest x-y (most negative)

    return rect


def correct_perspective():
    global selected_points, original_image, image_path
    
    if len(selected_points) != 4:
        print("错误：必须选择4个点才能进行校正。")
        return

    # 1. 选点转float32并自动排序
    pts_src = np.float32(selected_points)
    pts_src = order_points(pts_src)
    print(f"排序后的点坐标 (左上->右上->右下->左下): \n{pts_src}")

    # 2. A4纸比例：210x297mm -> 宽:高 = 210:297
    # 设定输出宽度，根据A4比例计算高度
    target_width = 2100 # 可以根据需要调整
    target_height = int(target_width * (297 / 210))
    print(f"目标输出尺寸: {target_width} x {target_height}")

    # 3. 目标点，和排序后的点一一对应
    pts_dst = np.float32([
        [0, 0],                      # 左上
        [target_width, 0],           # 右上
        [target_width, target_height], # 右下
        [0, target_height]           # 左下
    ])

    # 4. 透视变换矩阵与校正
    try:
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        corrected_image = cv2.warpPerspective(original_image, matrix, (target_width, target_height))
    except cv2.error as e:
        print(f"OpenCV错误：透视变换失败。可能的原因是选点不构成凸四边形或共线。错误详情: {e}")
        return

    # 5. 显示和保存
    cv2.imshow('Corrected Document', corrected_image)
    
    name, ext = os.path.splitext(image_path)
    output_file_name = f"{name}_corrected{ext}"
    success = cv2.imwrite(output_file_name, corrected_image)
    if success:
        print(f"\n校正完成！图像已保存为: {output_file_name}")
    else:
        print(f"\n错误：无法保存校正后的图像到: {output_file_name}")

def main():
    global selected_points, original_image, image_path

    # ========== 在这里修改你的图片路径 ==========
    image_path = '/home/alexander/cv-course/project5/text.png'

    # 加载图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误：无法加载图像 '{image_path}'。请检查路径是否正确。")
        return
    
    # 重置点列表
    selected_points = []

    # 创建窗口并设置鼠标回调
    window_name = 'Select 4 Corners of Document'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600) # 设置初始窗口大小
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- 透视校正工具 ---")
    print(f"加载图像: {image_path}")
    print("请按任意顺序点击文档的四个角点（程序会自动排序）：")
    print("点击完成后，校正将自动开始。")
    print("按 'r' 键可以重置所有已选点，按 'ESC' 键退出程序。")

    # 显示初始图像
    cv2.imshow(window_name, original_image)
    
    # 主循环，等待用户操作或按键
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            print("\n用户中断，退出程序。")
            break
        elif key == ord('r'):  # 按'r'重置
            selected_points = []
            print("已重置所有点，请重新选择。")
            cv2.imshow(window_name, original_image)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()