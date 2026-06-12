import os
import json
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

# ===================== 配置参数 =====================
VIDEO_ROOT = "/home/alexander/cv-course/homework11/badminton_storke_video"  # 数据集根目录
SAVE_DIR = "./data"
TARGET_FRAMES = 30       # 统一帧数 T=30
KEYPOINT_NUM = 33        # MediaPipe Pose 33个关键点
FEATURE_DIM = KEYPOINT_NUM * 4  # 132维 (x,y,z,visibility)
TEST_RATIO = 0.2

# 类别映射（与实验文档一致）
LABEL_MAP = {
    0: "forehand drive",
    1: "forehand lift",
    2: "forehand net shot",
    3: "forehand clear",
    4: "backhand drive",
    5: "backhand net shot"
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 创建保存文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

# ===================== 工具函数 =====================
def extract_pose_from_frame(frame):
    """单帧提取33个关键点 (x,y,z,visibility)，返回132维向量"""
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    keypoints = np.zeros((KEYPOINT_NUM, 4), dtype=np.float32)
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[idx] = [lm.x, lm.y, lm.z, lm.visibility]
    return keypoints.flatten()  # 展平为 132维

def resample_frames(seq, target_len):
    """不等长帧序列重采样为固定帧数"""
    n = len(seq)
    if n == target_len:
        return seq
    # 等间隔采样
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return seq[indices]

def normalize_skeleton(seq):
    """骨架归一化：以左右髋中点为原点，肩宽做尺度归一化"""
    for i in range(TARGET_FRAMES):
        frame = seq[i].reshape(KEYPOINT_NUM, 4)
        # 左右髋关键点编号：23(left hip), 24(right hip)
        hip_l = frame[23, :2]
        hip_r = frame[24, :2]
        hip_center = (hip_l + hip_r) / 2.0

        # 左右肩：11(left shoulder),12(right shoulder)
        shoulder_l = frame[11, :2]
        shoulder_r = frame[12, :2]
        shoulder_width = np.linalg.norm(shoulder_l - shoulder_r)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0

        # 平移+缩放
        frame[:, :2] = (frame[:, :2] - hip_center) / shoulder_width
        seq[i] = frame.flatten()
    return seq

# ===================== 遍历数据集处理 =====================
all_data = []
all_labels = []

# 遍历类别文件夹（文件夹名=类别英文）
for cls_name, label in REV_LABEL_MAP.items():
    cls_dir = os.path.join(VIDEO_ROOT, cls_name)
    if not os.path.exists(cls_dir):
        print(f"警告：不存在文件夹 {cls_dir}，跳过")
        continue
    print(f"正在处理类别: {cls_name} (标签{label})")

    # 遍历视频文件
    for vid_name in os.listdir(cls_dir):
        vid_path = os.path.join(cls_dir, vid_name)
        if not vid_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        cap = cv2.VideoCapture(vid_path)
        frame_seq = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            feat = extract_pose_from_frame(frame)
            frame_seq.append(feat)
        cap.release()

        if len(frame_seq) < 5:
            continue  # 跳过过短视频

        # 1. 重采样到30帧 2. 归一化
        frame_seq = np.array(frame_seq, dtype=np.float32)
        frame_seq = resample_frames(frame_seq, TARGET_FRAMES)
        frame_seq = normalize_skeleton(frame_seq)

        all_data.append(frame_seq)
        all_labels.append(label)

# 转为numpy数组
all_data = np.array(all_data, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)
print(f"总样本数: {len(all_data)}, 数据shape: {all_data.shape}")

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    all_data, all_labels, test_size=TEST_RATIO, random_state=42, stratify=all_labels
)

# 保存数据
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

# 保存标签映射
with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

print("数据预处理完成！文件已保存至 ./data 目录")
pose.close()