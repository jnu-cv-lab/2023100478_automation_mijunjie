import cv2
import numpy as np
import time

# ===================== 路径 =====================
path1 = "/home/alexander/cv-course/project6/box.png"
path2 = "/home/alexander/cv-course/project6/box_in_scene.png"

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
img2_color = cv2.imread(path2)

# ===================== 通用函数 =====================
def get_matches_orb(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return kp1, kp2, matches, des1, des2

def get_matches_sift(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe ratio test
            good.append(m)
    return kp1, kp2, good, des1, des2

def ransac_and_localize(img1, img2_color, kp1, kp2, matches):
    if len(matches) < 4:
        return None, 0, 0, 0, False
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = sum(mask.ravel())
    ratio = inliers / len(matches)
    try:
        h, w = img1.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(corners, H)
        loc_success = True
    except:
        loc_success = False
    return H, len(matches), inliers, ratio, loc_success

# ===================== 测试 ORB =====================
start = time.time()
kp1_orb, kp2_orb, matches_orb, _, _ = get_matches_orb(img1, img2)
H_orb, match_orb, in_orb, ratio_orb, succ_orb = ransac_and_localize(img1, img2_color, kp1_orb, kp2_orb, matches_orb)
time_orb = time.time() - start

# ===================== 测试 SIFT =====================
start = time.time()
kp1_sift, kp2_sift, matches_sift, _, _ = get_matches_sift(img1, img2)
H_sift, match_sift, in_sift, ratio_sift, succ_sift = ransac_and_localize(img1, img2_color, kp1_sift, kp2_sift, matches_sift)
time_sift = time.time() - start

# ===================== 绘制 SIFT 定位图 =====================
if H_sift is not None:
    h, w = img1.shape
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(corners, H_sift)
    img_sift_result = cv2.polylines(img2_color.copy(), [np.int32(dst)], True, (0,255,0), 3)
    cv2.imwrite('/home/alexander/cv-course/project6/sift_detection_result.png', img_sift_result)

# ===================== 输出对比表 =====================
print("==================== SIFT vs ORB 对比实验 ====================")
print(f"{'方法':<8}{'匹配数量':<10}{'内点数':<10}{'内点比例':<12}{'定位成功':<10}{'运行速度'}")
print("-" * 65)
print(f"{'ORB':<8}{match_orb:<10}{in_orb:<10}{ratio_orb:<12.2%}{succ_orb:<10}{'快 (实时)'}")
print(f"{'SIFT':<8}{match_sift:<10}{in_sift:<10}{ratio_sift:<12.2%}{succ_sift:<10}{'较慢'}")
print("-" * 65)