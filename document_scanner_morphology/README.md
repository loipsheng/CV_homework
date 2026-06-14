# 基于形态学增强的文档图像智能扫描与矫正系统

本项目是计算机视觉课程的传统图像处理项目，目标是对手机拍摄的文档图像进行自动边界定位、透视矫正、阴影去除、形态学增强和二值化输出。项目严格不使用深度学习框架或预训练模型，所有核心算法均基于 OpenCV 基础函数和 NumPy 实现。

## 项目结构

```text
document_scanner_morphology/
├── data/raw/                  # 放置手机拍摄的原始文档图像
├── data/samples/              # 可放置示例图
├── outputs/                   # 自动生成的中间结果和最终结果
├── src/                       # 核心算法模块
├── experiments/               # 批量运行和消融实验
├── report/figures/            # 报告可用图
├── main.py                    # 主程序入口
├── requirements.txt
└── README.md
```

## 环境配置

建议使用 Python 3.9 或以上版本。

```bash
pip install -r requirements.txt
```

依赖只包含 `opencv-python` 和 `numpy`。

## 数据准备

将手机拍摄的文档图像放入 `data/raw/`，支持 `jpg`、`jpeg`、`png`、`bmp`。建议至少采集 20 张，包括 A4 文档、实验报告、试卷、书页、有阴影文档、倾斜文档和复杂背景文档。

## 运行方式

```bash
python main.py --input_dir data/raw --output_dir outputs
python experiments/ablation.py --input_dir data/raw --output_dir outputs/ablation
python experiments/run_all.py
```

如果 `data/raw/` 为空，程序会给出提示并生成空的 `outputs/metrics.csv`，不会崩溃。

## 算法流程

1. 灰度化：将 BGR 图像转换为单通道灰度图。
2. 高斯滤波：抑制噪声，减少 Canny 边缘检测中的伪边缘。
3. Canny 边缘检测：提取文档可能的边缘结构。
4. 轮廓检测与多边形近似：使用 `findContours` 与 `approxPolyDP` 定位文档四角。
5. 四点透视变换：将倾斜四边形文档映射为正视矩形。
6. 形态学阴影去除：使用闭运算估计背景光照并归一化。
7. 顶帽/底帽增强：增强文字细节并改善背景均匀性。
8. 二值化对比：实现固定阈值、Otsu 和手写 Sauvola 局部阈值。
9. 二值后处理：开运算去除小噪点，闭运算连接断裂笔画。

## 输出说明

- `outputs/01_gray/`：灰度图
- `outputs/02_blur/`：高斯滤波结果
- `outputs/03_edges/`：Canny 边缘图
- `outputs/04_contours/`：文档轮廓和角点可视化
- `outputs/05_warped/`：透视矫正结果
- `outputs/06_shadow_removed/`：阴影去除结果
- `outputs/07_enhanced/`：形态学增强结果
- `outputs/08_binary_fixed/`：固定阈值二值图
- `outputs/09_binary_otsu/`：Otsu 二值图
- `outputs/10_binary_sauvola/`：Sauvola 二值图
- `outputs/comparisons/`：横向拼接对比图
- `outputs/metrics.csv`：无监督质量指标
- `outputs/ablation/ablation_results.csv`：参数消融实验指标

## 评价指标

- `contrast`：灰度标准差，反映图像对比度。
- `sharpness`：Laplacian 方差，反映图像清晰度。
- `foreground_ratio`：二值图中黑色文字像素比例，用于判断二值化是否过度或不足。
- `edge_density`：Canny 边缘像素比例，用于分析结构细节保留情况。

## 参数说明

- `Canny low/high threshold`：控制边缘检测敏感度，阈值低会保留更多弱边缘，也可能增加噪声。
- `morphology kernel_size`：控制形态学结构元素大小，过小会导致阴影估计不足，过大可能损失局部文字细节。
- `Sauvola window_size`：局部统计窗口大小，影响局部阈值对光照变化的适应范围。
- `Sauvola k`：控制局部标准差对阈值的影响，数值越大对局部对比度越敏感。

## 项目结论

Sauvola 局部阈值更适合阴影和光照不均的文档；Otsu 在光照较均匀场景下表现较好；固定阈值鲁棒性较差。形态学阴影去除和顶帽/底帽增强能够改善背景均匀性和文字对比度。若老师需要复现实验，请先将图片放入 `data/raw/`，再运行上述命令。本项目不包含任何深度学习模型或预训练模型。

