import cv2
import mediapipe as mp
import numpy as np
import torch
import json

# --------------------------
# 直接在这里定义所有常量，不再从train.py导入
# --------------------------
KEYPOINT_NUM = 33          # MediaPipe Pose 关键点数量
FEATURE_DIM = KEYPOINT_NUM * 4  # 每个关键点有 x,y,z,visibility 4个维度
TARGET_FRAMES = 30         # 预处理时固定的帧数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 只从train.py导入模型类，不导入其他变量
from train import SkeletonTransformer


# ===================== 配置 =====================
VIDEO_PATH = "/home/alexander/cv-course/homework11/demo.mp4"  # 待推理视频路径
MODEL_PATH = "/home/alexander/cv-course/homework11/badminton_transformer.pth"
DATA_DIR = "/home/alexander/cv-course/homework11/data"

# 加载标签映射
with open(f"{DATA_DIR}/label_map.json", "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
LABEL_MAP = {int(k): v for k, v in LABEL_MAP.items()}

# 初始化MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# ===================== 复用预处理函数 =====================
def extract_pose_from_frame(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    keypoints = np.zeros((KEYPOINT_NUM, 4), dtype=np.float32)
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[idx] = [lm.x, lm.y, lm.z, lm.visibility]
    return keypoints.flatten()

def resample_frames(seq, target_len):
    n = len(seq)
    if n == target_len:
        return seq
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return seq[indices]

def normalize_skeleton(seq):
    for i in range(TARGET_FRAMES):
        frame = seq[i].reshape(KEYPOINT_NUM, 4)
        hip_l = frame[23, :2]
        hip_r = frame[24, :2]
        hip_center = (hip_l + hip_r) / 2.0

        shoulder_l = frame[11, :2]
        shoulder_r = frame[12, :2]
        shoulder_width = np.linalg.norm(shoulder_l - shoulder_r)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0

        frame[:, :2] = (frame[:, :2] - hip_center) / shoulder_width
        seq[i] = frame.flatten()
    return seq

# ===================== 视频转骨架序列 =====================
def video2skeleton(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frame_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feat = extract_pose_from_frame(frame)
        frame_seq.append(feat)
    cap.release()

    frame_seq = np.array(frame_seq, dtype=np.float32)
    frame_seq = resample_frames(frame_seq, TARGET_FRAMES)
    frame_seq = normalize_skeleton(frame_seq)
    return frame_seq

# ===================== 推理主函数 =====================
def inference():
    # 加载模型
    model = SkeletonTransformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 视频转骨架
    skeleton_seq = video2skeleton(VIDEO_PATH)
    # 增加batch维度 [30,132] -> [1,30,132]
    input_tensor = torch.from_numpy(skeleton_seq).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_idx].item()

    # 输出结果
    pred_class = LABEL_MAP[pred_idx]
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {conf:.2f}")
    pose.close()

if __name__ == "__main__":
    inference()