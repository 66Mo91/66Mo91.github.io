# MLP 多层感知机 · 学习参数实现与可视化（课程作业）

本作业用 **Python + NumPy 从零手写** 一个两层隐藏层的多层感知机（MLP），
完成前向传播、反向传播（链式法则逐层推导）与全批量梯度下降训练，
并在 two-moons（双月牙）二分类玩具数据上对**学习参数**与训练过程进行可视化，
最终结果展示在可离线打开的网页中。

> 未使用 PyTorch / TensorFlow 等深度学习框架的训练封装；激活函数、损失函数、
> 反向梯度与参数更新均在 `mlp_train.py` 中手写并附中文注释。

## 一、目录结构

```
38453242849117442/
├── mlp_train.py        # 主训练脚本：手写 MLP 训练，导出 data.js 与静态图
├── index.html          # 离线可视化页面（内联 CSS/JS，原生 Canvas 绘图，无外网 CDN）
├── requirements.txt    # 依赖清单（仅 numpy、matplotlib）
├── README.md           # 本说明文件
├── data/
│   └── data.js         # 训练导出的数据契约：window.MLP_DATA = {...}
└── assets/             # matplotlib 生成的静态备份图（PNG）
    ├── curves_loss_acc.png     # 损失 / 准确率曲线
    ├── curves_norm.png         # 权重范数 / 梯度范数曲线
    ├── weights_heatmap.png     # 最终各层权重热图
    └── decision_boundary.png   # 决策边界训练前 / 中 / 后对比
```

## 二、环境依赖与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 等价于：
# pip install numpy==1.26.4 matplotlib==3.11.0
```

### 2. 复现训练并重新导出数据与图表

```bash
python3 mlp_train.py
```

脚本会：

1. 固定随机种子（`SEED = 42`）手写生成 two-moons 数据；
2. 训练网络并在终端打印**最终准确率、最终损失、各层权重形状**；
3. 重新生成 `data/data.js` 与 `assets/` 下的四张静态图。

### 3. 查看网页

直接用浏览器双击打开 `index.html` 即可（页面通过 `<script src="data/data.js">`
读取本地数据，不依赖外网 CDN；图表由原生 Canvas 绘制）。

> 若直接双击后图表区域显示“内置演示数据”，说明 `data/data.js` 未被加载，
> 请先运行一次 `python3 mlp_train.py`。

## 三、模型与方法

- **网络结构**：输入层 2 维 → 隐藏层 1（8，tanh）→ 隐藏层 2（8，tanh）→ 输出层（2，softmax）
- **隐藏激活**：tanh；**输出层**：softmax
- **损失函数**：交叉熵（Cross Entropy）
- **优化器**：全批量梯度下降（SGD），学习率 0.2，训练 1500 轮
- **参数初始化**：Xavier 初始化

反向传播要点（详见 `mlp_train.py` 中文注释与网页“方法与公式说明”）：

- 输出层（softmax + 交叉熵）梯度：`dZ3 = A3 − Y`
- 隐藏层反传：`dZ = (dZ_next · W_nextᵀ) ⊙ (1 − A²)`
- 权重梯度：`dW = A_prevᵀ · dZ / N`；偏置梯度：`db = mean(dZ, 轴=0)`
- 参数更新：`W ← W − η·dW`，`b ← b − η·db`

## 四、可视化内容

网页与静态图包含：

- **决策边界演化**：训练前 / 训练中 / 训练后三个阶段的分类区域对比；
- **训练曲线**：交叉熵损失、训练准确率、权重范数 ‖W‖、梯度范数 ‖dW‖；
- **权重参数热图**：最终 W1 / W2 / W3 权重矩阵（红正蓝负）与偏置条形图；
- **权重快照**：第一层 W1 在训练前 / 中 / 后的热图对比（同一色标）；
- **结构示意**：SVG 绘制的网络层级与全连接示意。

## 五、部署到自己的网站

本页面为纯静态站点，部署方式与普通网页一致：

1. 保持目录相对结构不变，把整个文件夹上传到网站服务器 / 静态托管（如 GitHub Pages、
   个人服务器、对象存储静态网站等）：
   - `index.html` 在根目录；
   - `data/data.js`、`assets/*.png` 的相对路径保持不变。
2. 访问 `index.html` 即可；页面不加载任何外部脚本、字体或 CDN 资源，可离线运行。
3. 如需更新结果，在本地重新运行 `python3 mlp_train.py` 后，
   把新的 `data/` 与 `assets/` 一并上传覆盖即可。

## 六、本次训练结果（固定种子可复现）

| 指标 | 数值 |
|---|---|
| 最终准确率 final_acc | 1.0000（100.0%） |
| 最终损失 final_loss | 0.0009 |
| W1 形状 | (2, 8) |
| W2 形状 | (8, 8) |
| W3 形状 | (8, 2) |

## 七、备注

- 数据集由脚本内部手写生成（two-moons + 高斯噪声），不依赖 scikit-learn；
- 随机种子固定为 42，重复运行结果一致；
- 本作业聚焦 two-moons 玩具数据上的参数与边界可视化；MNIST 手写数字版本为可选扩展，本次未包含。

> （注：部分内容由豆包 AI 生成）