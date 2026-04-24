import cv2
import numpy as np

# ===================== 路径 =====================
path1 = "/home/alexander/cv-course/project6/box.png"
path2 = "/home/alexander/cv-course/project6/box_in_scene.png"

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
img2_color = cv2.imread(path2)

# 要测试的三组 nfeatures
nfeatures_list = [500, 1000, 2000]

print("==================== 任务6 参数对比实验 ====================")
print(f"{'nfeatures':<10}{'模板关键点':<10}{'场景关键点':<10}{'匹配数':<8}{'内点数':<8}{'内点比例':<10}{'定位成功'}")
print("-" * 70)

for nfeat in nfeatures_list:
    # ===================== ORB 特征检测 =====================
    orb = cv2.ORB_create(nfeatures=nfeat)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # ===================== 匹配 =====================
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    num_matches = len(matches)

    # ===================== RANSAC =====================
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    num_inliers = sum(mask.ravel())
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0

    # ===================== 定位判断 =====================
    h, w = img1.shape
    try:
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(corners, H)
        success = "是"
    except:
        success = "否"

    # ===================== 输出表格 =====================
    print(f"{nfeat:<10}{len(kp1):<10}{len(kp2):<10}{num_matches:<8}{num_inliers:<8}{inlier_ratio:<10.2%}{success}")

print("\n==================== 实验结束 ====================")