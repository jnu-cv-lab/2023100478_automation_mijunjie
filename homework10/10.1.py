import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. Sinusoidal Position Encoding (Absolute PE)
# ----------------------
def sinusoidal_pe(seq_len, d_model, base=10000):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(base) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# ----------------------
# 2. 2D Rotation (RoPE Basic Unit)
# ----------------------
def rotate_2d(x, pos, theta=0.1):
    angle = pos * theta
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([x[0] * cos - x[1] * sin, x[0] * sin + x[1] * cos])

# ----------------------
# 3. High-Dimensional RoPE
# ----------------------
def rope_forward(x, pos, base=10000):
    seq_len, d_model = x.shape
    assert d_model % 2 == 0, "d_model must be even"
    half_dim = d_model // 2
    freqs = 1.0 / (base ** (np.arange(0, half_dim, 2) / half_dim))
    angle = pos * freqs
    cos = np.cos(angle)
    sin = np.sin(angle)
    cos = np.repeat(cos, 2)
    sin = np.repeat(sin, 2)
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    x_rot = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return x_rot

# ----------------------
# 4. Compare E+pos vs RoPE
# ----------------------
def compare_e_rope(emb, pos):
    d_model = emb.shape[1]
    pe = sinusoidal_pe(1, d_model)
    e_add = emb + pe
    rope_out = rope_forward(emb, pos)
    return e_add, rope_out

# ----------------------
# 5. Verify Relative Position Property【修复：去掉score[0]】
# ----------------------
def verify_relative_pos():
    print("\n=== Verifying RoPE Relative Position Property ===")
    q = np.array([[1, 0]])
    k = np.array([[1, 0]])
    theta = 0.1
    deltas = [1, 2, 3]
    pos_q_list = [0, 1, 2, 3, 4, 5]
    scores = {d: [] for d in deltas}
    for pos_q in pos_q_list:
        for delta in deltas:
            pos_k = pos_q + delta
            q_rot = rotate_2d(q[0], pos_q, theta)
            k_rot = rotate_2d(k[0], pos_k, theta)
            score = np.dot(q_rot, k_rot) # 结果是单个数字
            scores[delta].append(score)
            print(f"pos_q={pos_q}, delta={delta}, score={score:.4f}")
    return scores

# ----------------------
# Plot 1: RoPE Relative Position Score
# ----------------------
def plot_rope_relative():
    scores = verify_relative_pos()
    pos_q_list = [0, 1, 2, 3, 4, 5]
    plt.figure(figsize=(8, 4))
    for delta, s in scores.items():
        plt.plot(pos_q_list, s, marker='o', label=f"Δpos={delta}")
    plt.title("RoPE: Attention Score vs Query pos_q (fixed Δpos)")
    plt.xlabel("pos_q of Query")
    plt.ylabel("Attention Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("rope_relative_score.png")
    plt.show()

# ----------------------
# Plot 2: E+pos vs RoPE Vector Comparison
# ----------------------
def plot_feature_compare():
    emb = np.array([[1, 1]])
    pos = 2
    pe = sinusoidal_pe(1, 2)
    e_add = emb + pe
    rope_out = rope_forward(emb, pos)
    plt.figure(figsize=(6, 6))
    plt.quiver([0, 0, 0], [0, 0, 0],
               [emb[0, 0], e_add[0, 0], rope_out[0, 0]],
               [emb[0, 1], e_add[0, 1], rope_out[0, 1]],
               angles='xy', scale_units='xy', scale=1,
               color=['blue', 'red', 'green'])
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.title("Vector Comparison: E+pos vs RoPE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(["Original","E+pos","RoPE"])
    plt.grid(True)
    plt.savefig("pos_emb_compare.png")
    plt.show()

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    # 1. Print Sinusoidal PE
    pe = sinusoidal_pe(seq_len=5, d_model=8)
    print("=== 1. Sinusoidal Position Encoding (PE) ===")
    print(pe)

    # 2. Print 2D Rotation
    test_2d = np.array([1, 0])
    rot_2d = rotate_2d(test_2d, pos=3)
    print("\n=== 2. 2D Rotation ===")
    print(f"Original: {test_2d}, Rotated: {rot_2d}")

    # 3. Print High-Dim RoPE
    emb_test = np.random.randn(3, 8)
    rope_out = rope_forward(emb_test, pos=4)
    print("\n=== 3. High-Dimensional RoPE ===")
    print("Input:\n", emb_test)
    print("RoPE Output:\n", rope_out)

    # 4. Print E+pos vs RoPE
    emb = np.array([[1, 1, 2, 2]])
    e_add, rope_4 = compare_e_rope(emb, pos=2)
    print("\n=== 4. E+pos vs RoPE ===")
    print("E+pos Result:", e_add)
    print("RoPE Result:", rope_4)

    # 5. Plot RoPE Relative Position
    plot_rope_relative()

    # 6. Plot Vector Comparison
    plot_feature_compare()