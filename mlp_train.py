# -*- coding: utf-8 -*-
"""
MLP 多层感知机 —— 纯 NumPy 从零手写实现（课程作业）
=====================================================

网络结构：2 -> 8 -> 8 -> 2（输入层 2 维，两个隐藏层各 8 个神经元，输出 2 类）
激活函数：隐藏层 tanh，输出层 softmax
损失函数：交叉熵（Cross Entropy）
优化器：全批量梯度下降（SGD, full-batch）

本脚本不使用 PyTorch / TensorFlow 等深度学习框架的训练封装，
前向传播、反向传播（链式法则逐层推导）、参数更新全部手写并加中文注释。

运行方式：
    python3 mlp_train.py

运行后会：
    1. 在 data/data.js 中导出网页可视化所需的全部数据（window.MLP_DATA = {...}）；
    2. 在 assets/ 目录下生成静态备份图（损失/准确率曲线、权重热图、决策边界）；
    3. 在终端打印最终准确率、最终损失、各层权重形状。

依赖：numpy、matplotlib（见 requirements.txt）。
数据集：脚本内部手写生成 two-moons（双月牙）玩具数据，不依赖 sklearn。
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无图形界面环境下也能保存图片
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. 全局配置（固定随机种子，保证结果可复现）
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# 网络结构：输入 2 维 -> 隐藏层 8 -> 隐藏层 8 -> 输出 2 类
INPUT_DIM = 2
HIDDEN1 = 8
HIDDEN2 = 8
OUTPUT_DIM = 2

# 训练超参数
EPOCHS = 1500          # 训练轮数
LEARNING_RATE = 0.2    # 学习率
N_SAMPLES = 300        # 样本数量
NOISE = 0.18           # 月牙数据上叠加的高斯噪声标准差

# 历史记录采样间隔（每隔多少轮记录一次损失/准确率/范数）
RECORD_EVERY = 20

# 决策边界网格分辨率
GRID_NX = 120
GRID_NY = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSET_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. 手写 two-moons（双月牙）数据集生成
# ----------------------------------------------------------------------
def make_two_moons(n_samples=300, noise=0.18, seed=0):
    """手写生成 two-moons 二分类数据集（不依赖 sklearn）。

    上月牙（类别 0）：圆心在原点、半径 1 的上半圆弧；
    下月牙（类别 1）：向右平移 0.5、向下平移 0.5 的下半圆弧；
    最后在坐标上叠加高斯噪声，让分类任务更接近真实场景。

    参数:
        n_samples: 样本总数
        noise: 高斯噪声标准差
        seed: 随机种子
    返回:
        X: 形状 (n_samples, 2) 的坐标矩阵
        y: 形状 (n_samples,) 的标签，取值 0 / 1
    """
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    # 上月牙：角度 0 ~ pi，单位圆上半部分
    theta0 = rng.uniform(0.0, np.pi, n_half)
    x0 = np.cos(theta0)
    y0 = np.sin(theta0)

    # 下月牙：角度 0 ~ pi，关于 x 轴翻转后向右、向下平移，形成交错月牙
    theta1 = rng.uniform(0.0, np.pi, n_samples - n_half)
    x1 = 1.0 - np.cos(theta1)
    y1 = -0.5 - np.sin(theta1)

    X = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1]),
    ])
    y = np.array([0] * n_half + [1] * (n_samples - n_half))

    # 叠加高斯噪声并打乱顺序
    X += rng.normal(0.0, noise, X.shape)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    return X, y


# ----------------------------------------------------------------------
# 2. 激活函数及其导数（手写）
# ----------------------------------------------------------------------
def tanh(z):
    """tanh 激活函数。"""
    return np.tanh(z)


def tanh_grad(a):
    """tanh 的导数：d(tanh)/dz = 1 - tanh(z)^2。
    注意这里传入的是 tanh 的输出 a，直接用 a 计算即可。
    """
    return 1.0 - a ** 2


def softmax(z):
    """输出层 softmax：把线性分数归一化为各类别的概率分布。

    为防止指数上溢，先减去每行最大值（softmax 平移不变）。
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ----------------------------------------------------------------------
# 3. 参数初始化（Xavier 初始化，适合 tanh）
# ----------------------------------------------------------------------
def init_parameters():
    """初始化三层权重矩阵 W 和偏置 b。

    权重形状：
        W1: (2, 8)   输入层 -> 第一隐藏层
        W2: (8, 8)   第一隐藏层 -> 第二隐藏层
        W3: (8, 2)   第二隐藏层 -> 输出层
    偏置形状：
        b1: (8,)，b2: (8,)，b3: (2,)
    使用 Xavier 初始化：W ~ N(0, sqrt(2/(fan_in+fan_out)))。
    """
    params = {}

    def xavier(n_in, n_out):
        std = np.sqrt(2.0 / (n_in + n_out))
        return rng_normal(n_in, n_out, std)

    params["W1"] = xavier(INPUT_DIM, HIDDEN1)
    params["b1"] = np.zeros(HIDDEN1)
    params["W2"] = xavier(HIDDEN1, HIDDEN2)
    params["b2"] = np.zeros(HIDDEN2)
    params["W3"] = xavier(HIDDEN2, OUTPUT_DIM)
    params["b3"] = np.zeros(OUTPUT_DIM)
    return params


def rng_normal(n_in, n_out, std):
    """独立小工具：按给定标准差生成正态分布权重矩阵。"""
    return np.random.normal(0.0, std, size=(n_in, n_out))


# ----------------------------------------------------------------------
# 4. 前向传播（手写）
# ----------------------------------------------------------------------
def forward(X, params):
    """前向传播：逐层计算线性输出 Z 与激活输出 A。

    缓存中间量供反向传播使用：
        Z1, A1 = 第一隐藏层线性分数 / tanh 输出
        Z2, A2 = 第二隐藏层线性分数 / tanh 输出
        Z3, A3 = 输出层线性分数 / softmax 概率
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    # 第一隐藏层：Z1 = X·W1 + b1，A1 = tanh(Z1)
    Z1 = X @ W1 + b1
    A1 = tanh(Z1)

    # 第二隐藏层：Z2 = A1·W2 + b2，A2 = tanh(Z2)
    Z2 = A1 @ W2 + b2
    A2 = tanh(Z2)

    # 输出层：Z3 = A2·W3 + b3，A3 = softmax(Z3)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2,
             "Z3": Z3, "A3": A3}
    return A3, cache


def cross_entropy_loss(probs, y):
    """交叉熵损失（多分类形式，二分类同理）。

    probs: softmax 概率矩阵 (N, 2)
    y:     整数标签数组 (N,)，取值 0/1
    返回:  平均每条样本的交叉熵损失。
    """
    n = len(y)
    # 取出每个样本真实类别对应的概率，加极小值防止取 log(0)
    p_true = probs[np.arange(n), y]
    p_true = np.clip(p_true, 1e-12, 1.0)
    loss = -np.mean(np.log(p_true))
    return loss


def predict(X, params):
    """预测类别：取 softmax 概率最大的类别。"""
    probs, _ = forward(X, params)
    return np.argmax(probs, axis=1)


def accuracy(X, y, params):
    """计算分类准确率。"""
    pred = predict(X, params)
    return float(np.mean(pred == y))


# ----------------------------------------------------------------------
# 5. 反向传播（手写链式法则）
# ----------------------------------------------------------------------
def backward(params, cache, y):
    """根据缓存的前向中间量，反向逐层计算各参数梯度。

    记号约定：dZ 表示损失对该层线性分数 Z 的偏导数。

    推导要点：
      - 输出层 softmax + 交叉熵有简洁梯度：dZ3 = A3 - Y（Y 为 one-hot）
      - 隐藏层 tanh 反传：dZ = dA * tanh_grad(A)
      - 权重梯度：dW = 上一层激活的转置 @ dZ；偏置梯度：db = dZ 按样本求平均
      - 继续向上反传：dA_prev = dZ @ W 的转置
    """
    n = len(y)
    W1, W2, W3 = params["W1"], params["W2"], params["W3"]
    X, A1, A2, A3 = cache["X"], cache["A1"], cache["A2"], cache["A3"]

    # 把整数标签转成 one-hot 矩阵 (N, 2)
    Y_onehot = np.zeros_like(A3)
    Y_onehot[np.arange(n), y] = 1.0

    # --- 输出层梯度 ---
    # softmax + 交叉熵对 Z3 的导数：A3 - Y
    dZ3 = A3 - Y_onehot
    dW3 = A2.T @ dZ3 / n
    db3 = np.mean(dZ3, axis=0)

    # --- 第二隐藏层梯度 ---
    dA2 = dZ3 @ W3.T                       # 损失对 A2 的偏导
    dZ2 = dA2 * tanh_grad(A2)              # 乘上 tanh 导数
    dW2 = A1.T @ dZ2 / n
    db2 = np.mean(dZ2, axis=0)

    # --- 第一隐藏层梯度 ---
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * tanh_grad(A1)
    dW1 = X.T @ dZ1 / n
    db1 = np.mean(dZ1, axis=0)

    grads = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3,
    }
    return grads


def update_parameters(params, grads, lr):
    """梯度下降更新参数：参数 = 参数 - 学习率 * 梯度。"""
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        params[key] -= lr * grads["d" + key]
    return params


# ----------------------------------------------------------------------
# 6. 决策边界网格预测（用于可视化分类区域）
# ----------------------------------------------------------------------
def decision_grid(params, xmin, xmax, ymin, ymax, nx=120, ny=100):
    """在二维网格上预测类别，返回网格坐标与逐点类别（行优先展平）。"""
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    preds = predict(grid, params)
    return {
        "nx": nx,
        "ny": ny,
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "Z": preds.tolist(),
    }


def weight_norm(params):
    """所有层权重矩阵的 Frobenius 范数平方和的平方根（权重总范数）。"""
    total = 0.0
    for key in ("W1", "W2", "W3"):
        total += np.sum(params[key] ** 2)
    return float(np.sqrt(total))


def grad_norm(grads):
    """所有权重梯度的总范数（监控梯度是否消失/爆炸）。"""
    total = 0.0
    for key in ("dW1", "dW2", "dW3"):
        total += np.sum(grads[key] ** 2)
    return float(np.sqrt(total))


# ----------------------------------------------------------------------
# 7. 主训练流程
# ----------------------------------------------------------------------
def main():
    # 7.1 生成数据集
    X, y = make_two_moons(N_SAMPLES, NOISE, seed=SEED)

    # 网格范围（根据数据坐标自适应预留边距）
    xmin, xmax = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    ymin, ymax = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4

    # 7.2 初始化参数
    params = init_parameters()

    # 7.3 训练前快照（epoch = 0）
    pre_boundary = decision_grid(params, xmin, xmax, ymin, ymax,
                                 GRID_NX, GRID_NY)
    w1_initial = params["W1"].copy()

    # 7.4 训练历史记录
    history = {"epochs": [], "loss": [], "acc": [], "w_norm": [], "grad_norm": []}

    def record(epoch, loss, acc, params, grads):
        history["epochs"].append(int(epoch))
        history["loss"].append(float(loss))
        history["acc"].append(float(acc))
        history["w_norm"].append(weight_norm(params))
        history["grad_norm"].append(grad_norm(grads))

    # 先记录训练前的一次状态（用初始参数做一次前向 + 反向得到初始梯度范数）
    probs, cache0 = forward(X, params)
    loss0 = cross_entropy_loss(probs, y)
    acc0 = accuracy(X, y, params)
    grads0 = backward(params, cache0, y)
    record(0, loss0, acc0, params, grads0)

    mid_epoch = EPOCHS // 2
    mid_snapshot = None
    w1_mid = None

    # 7.5 梯度下降主循环
    for epoch in range(1, EPOCHS + 1):
        probs, cache = forward(X, params)
        loss = cross_entropy_loss(probs, y)
        acc = accuracy(X, y, params)
        grads = backward(params, cache, y)
        params = update_parameters(params, grads, LEARNING_RATE)

        if epoch % RECORD_EVERY == 0 or epoch == EPOCHS:
            record(epoch, loss, acc, params, grads)

        # 训练中（中点）快照
        if epoch == mid_epoch:
            mid_snapshot = decision_grid(params, xmin, xmax, ymin, ymax,
                                         GRID_NX, GRID_NY)
            w1_mid = params["W1"].copy()

    # 7.6 训练后快照
    final_boundary = decision_grid(params, xmin, xmax, ymin, ymax,
                                   GRID_NX, GRID_NY)
    w1_final = params["W1"].copy()

    final_probs, _ = forward(X, params)
    final_loss = cross_entropy_loss(final_probs, y)
    final_acc = accuracy(X, y, params)

    # 7.7 组装数据契约（data/data.js 严格使用以下字段）
    meta = {
        "dataset": "two-moons（手写双月牙二分类玩具数据）",
        "n_samples": int(N_SAMPLES),
        "hidden": [HIDDEN1, HIDDEN2],
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "final_loss": float(final_loss),
        "final_acc": float(final_acc),
        "activation": "tanh",
        "loss": "cross_entropy",
        "optimizer": "SGD",
    }

    points = {"X": X.tolist(), "y": y.tolist()}

    boundaries = [
        {"epoch": 0, "title": "训练前", "grid": pre_boundary},
        {"epoch": mid_epoch, "title": "训练中", "grid": mid_snapshot},
        {"epoch": EPOCHS, "title": "训练后", "grid": final_boundary},
    ]

    weights = {
        "layers": [
            {"name": "W1（输入层→隐藏层1）", "rows": int(params["W1"].shape[0]),
             "cols": int(params["W1"].shape[1]), "matrix": params["W1"].tolist()},
            {"name": "W2（隐藏层1→隐藏层2）", "rows": int(params["W2"].shape[0]),
             "cols": int(params["W2"].shape[1]), "matrix": params["W2"].tolist()},
            {"name": "W3（隐藏层2→输出层）", "rows": int(params["W3"].shape[0]),
             "cols": int(params["W3"].shape[1]), "matrix": params["W3"].tolist()},
        ],
        "biases": [
            {"name": "b1", "values": params["b1"].tolist()},
            {"name": "b2", "values": params["b2"].tolist()},
            {"name": "b3", "values": params["b3"].tolist()},
        ],
    }

    weight_snapshots = [
        {"epoch": 0, "layer": "W1", "matrix": w1_initial.tolist()},
        {"epoch": mid_epoch, "layer": "W1", "matrix": w1_mid.tolist()},
        {"epoch": EPOCHS, "layer": "W1", "matrix": w1_final.tolist()},
    ]

    data = {
        "meta": meta,
        "points": points,
        "history": history,
        "boundaries": boundaries,
        "weights": weights,
        "weight_snapshots": weight_snapshots,
    }

    # 7.8 导出 data.js（网页直接读取 window.MLP_DATA）
    js_content = "// 由 mlp_train.py 自动生成，请勿手工修改\n"
    js_content += "window.MLP_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
    js_path = os.path.join(DATA_DIR, "data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    # 7.9 生成静态备份图（matplotlib）
    plot_static_figures(X, y, params, history, boundaries)

    # 7.10 终端打印验收信息
    print("=" * 50)
    print("训练完成")
    print(f"最终准确率 final_acc = {final_acc:.4f}")
    print(f"最终损失   final_loss = {final_loss:.4f}")
    print("各层权重形状：")
    for key in ("W1", "W2", "W3"):
        print(f"  {key}: {params[key].shape}")
    print(f"数据已导出：{js_path}")
    print("=" * 50)


# ----------------------------------------------------------------------
# 8. matplotlib 静态备份图
# ----------------------------------------------------------------------
def plot_static_figures(X, y, params, history, boundaries):
    """生成静态 PNG 备份图：损失/准确率曲线、权重热图、决策边界对比。"""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    # 8.1 损失与准确率曲线
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epochs"], history["loss"], color="#d9534f")
    axes[0].set_title("训练损失（交叉熵）")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epochs"], history["acc"], color="#337ab7")
    axes[1].set_title("训练准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "curves_loss_acc.png"), dpi=120)
    plt.close(fig)

    # 8.2 权重范数 / 梯度范数曲线
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epochs"], history["w_norm"], color="#5cb85c")
    axes[0].set_title("权重范数 ||W||")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["epochs"], history["grad_norm"], color="#f0ad4e")
    axes[1].set_title("梯度范数 ||dW||")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "curves_norm.png"), dpi=120)
    plt.close(fig)

    # 8.3 最终各层权重热图
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, layer in zip(axes, ["W1", "W2", "W3"]):
        mat = params[layer]
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto")
        ax.set_title(layer)
        ax.set_xlabel("神经元")
        ax.set_ylabel("输入")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "weights_heatmap.png"), dpi=120)
    plt.close(fig)

    # 8.4 决策边界三阶段对比
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, snap in zip(axes, boundaries):
        g = snap["grid"]
        gx = np.linspace(g["xmin"], g["xmax"], g["nx"])
        gy = np.linspace(g["ymin"], g["ymax"], g["ny"])
        Z = np.array(g["Z"]).reshape(g["ny"], g["nx"])
        ax.contourf(gx, gy, Z, alpha=0.3, cmap="RdBu")
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, c="#1f77b4", edgecolor="k", linewidth=0.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, c="#d62728", edgecolor="k", linewidth=0.3)
        ax.set_title(snap["title"] + f"（epoch {snap['epoch']}）")
        ax.set_xlim(g["xmin"], g["xmax"])
        ax.set_ylim(g["ymin"], g["ymax"])
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, "decision_boundary.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
