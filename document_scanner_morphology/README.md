# 基于形态学增强的文档图像智能扫描与矫正系统

本项目是计算机视觉课程的传统图像处理项目，用于处理手机拍摄的文档图像。系统会自动完成文档角点定位、透视矫正、阴影去除、形态学增强和三种二值化方法对比。

本项目严格不使用深度学习框架或预训练模型，所有核心算法均基于 OpenCV 基础函数和 NumPy 实现。

## 一、项目整体流程

可以把本项目理解成一个简化版“手机扫描 App”。完整流程如下：

```text
手机拍摄文档图片
→ 放入 data/raw/
→ 批量读取图片
→ 缩放到合适尺寸
→ 灰度化
→ 高斯滤波去噪
→ Canny 边缘检测
→ 轮廓检测与多边形近似
→ 自动定位文档四个角点
→ 透视变换矫正倾斜文档
→ 阴影去除
→ 形态学文字增强
→ 固定阈值 / Otsu / Sauvola 二值化
→ 保存中间结果、最终结果、对比图和评价指标
```

对应课程任务：

1. 使用 Canny 边缘检测和轮廓多边形近似定位文档角点，再用透视变换完成倾斜校正。
2. 比较固定阈值、Otsu、Sauvola 三种二值化方法，并结合形态学运算优化前景文字区域。
3. 在自建手机拍摄文档集上进行全流程测试，展示角点检测、形态学增强、最终二值图像等中间结果。

## 二、项目结构与每个文件夹作用

```text
document_scanner_morphology/
├── data/
│   ├── raw/
│   ├── samples/
│   └── README.md
├── outputs/
│   ├── 01_gray/
│   ├── 02_blur/
│   ├── 03_edges/
│   ├── 04_contours/
│   ├── 05_warped/
│   ├── 06_shadow_removed/
│   ├── 07_enhanced/
│   ├── 08_binary_fixed/
│   ├── 09_binary_otsu/
│   ├── 10_binary_sauvola/
│   ├── comparisons/
│   ├── ablation/
│   └── metrics.csv
├── src/
├── experiments/
├── report/
│   ├── figures/
│   └── report.md
├── main.py
├── requirements.txt
├── FINAL_CHECKLIST.md
└── README.md
```

### `data/`

存放输入数据和数据说明。

- `data/raw/`：真正参与实验的原始手机拍摄文档图片。主程序默认读取这里。
- `data/samples/`：可选示例图片目录，当前主程序不会默认处理它。
- `data/README.md`：说明应该如何采集、命名和准备实验图片。

### `outputs/`

存放程序自动生成的所有中间结果、最终结果和评价表格。这个目录是你检查实验结果时最重要的地方。

### `src/`

存放核心算法代码。

- `io_utils.py`：图片读取、保存、批量列出、缩放。
- `preprocessing.py`：灰度化、高斯滤波、直方图均衡化。
- `document_detection.py`：Canny 边缘检测、轮廓查找、角点排序、角点可视化。
- `perspective.py`：四点透视变换，把斜拍文档拉正。
- `morphology_enhancement.py`：形态学阴影去除、顶帽/底帽增强。
- `binarization.py`：固定阈值、Otsu、Sauvola 二值化。
- `evaluation.py`：计算对比度、清晰度、前景比例、边缘密度。
- `visualization.py`：生成横向拼接图和报告图。
- `pipeline.py`：串联完整处理流程。

### `experiments/`

存放实验脚本。

- `ablation.py`：参数消融实验，比较不同 Canny 阈值、形态学核大小、Sauvola 参数的影响。
- `run_all.py`：重新跑主流程和消融实验，并把报告可用图片生成到 `report/figures/`。

### `report/`

存放课程报告相关内容。

- `report/report.md`：课程报告初稿。
- `report/figures/`：最终可插入报告的图片。

## 三、outputs 每个文件夹的作用和怎么看

### `outputs/01_gray/`

灰度图。原始彩色图片被转换成黑白灰图像。  
作用：大多数传统图像处理算法都先在灰度图上处理。

### `outputs/02_blur/`

高斯滤波后的图。  
作用：去掉一部分噪声，避免 Canny 边缘检测时出现太多杂乱边缘。

### `outputs/03_edges/`

Canny 边缘检测结果。白色线条表示检测到的边缘。  
作用：帮助程序找到纸张边界，也能展示边缘检测效果。  
报告用途：可用于说明 Canny 阈值对边缘数量的影响。

### `outputs/04_contours/`

文档轮廓和角点检测结果。图中会画出文档边框和角点。  
作用：验证程序是否正确找到了文档四个角。  
重点检查：绿色框是否贴合纸张边缘。如果框错了，后面的透视矫正也会受影响。

### `outputs/05_warped/`

透视矫正后的文档图。  
作用：把原来倾斜拍摄的文档拉正，变成类似扫描件的矩形。  
报告用途：用于展示“倾斜校正”效果。

### `outputs/06_shadow_removed/`

阴影去除结果。  
作用：通过形态学闭运算估计背景光照，让纸面亮度更均匀。  
重点检查：原图中偏暗的阴影区域是否变得更接近正常纸面。

### `outputs/07_enhanced/`

形态学增强结果。  
作用：结合阴影去除、顶帽、底帽变换，提高文字和背景之间的对比度。  
报告用途：用于展示形态学增强前后的区别。

### `outputs/08_binary_fixed/`

固定阈值二值化结果。  
作用：用一个固定阈值把图像分成黑色文字和白色背景。  
特点：简单，但对阴影和光照变化很敏感，鲁棒性较差。

### `outputs/09_binary_otsu/`

Otsu 二值化结果。  
作用：自动计算一个全局阈值。  
特点：光照比较均匀时效果较好；如果有明显阴影，全局阈值可能不够灵活。

### `outputs/10_binary_sauvola/`

Sauvola 二值化结果。  
作用：根据局部窗口的均值和标准差计算每个区域自己的阈值。  
特点：更适合阴影、光照不均的文档，但参数不合适时可能产生更多噪声。

### `outputs/comparisons/`

每张图片的完整流程横向对比图。  
通常包含：

```text
Original / Contour / Warped / Enhanced / Fixed / Otsu / Sauvola
```

作用：最适合快速检查整张图片从原图到最终二值图的全过程。  
报告用途：可以挑选效果最好的一两张作为流程展示图。

### `outputs/ablation/`

参数消融实验结果。  
作用：比较不同参数对结果的影响，例如：

- 不同 Canny 阈值对边缘检测的影响。
- 不同形态学核大小对阴影去除和增强效果的影响。
- 不同 Sauvola 参数对二值化结果的影响。

报告用途：用于写“参数分析”或“消融实验”。

### `outputs/metrics.csv`

自动生成的定量指标表格。  
字段含义：

- `enhanced_contrast`：增强图像的对比度，数值越大表示灰度变化越明显。
- `enhanced_sharpness`：增强图像的清晰度，用 Laplacian 方差衡量。
- `fixed_foreground_ratio`：固定阈值结果中黑色文字像素比例。
- `otsu_foreground_ratio`：Otsu 结果中黑色文字像素比例。
- `sauvola_foreground_ratio`：Sauvola 结果中黑色文字像素比例。
- `fixed_edge_density`：固定阈值结果中的边缘密度。
- `otsu_edge_density`：Otsu 结果中的边缘密度。
- `sauvola_edge_density`：Sauvola 结果中的边缘密度。

这些指标不代表绝对好坏，主要用于报告中做定量对比。

## 四、最终可用于报告的图片

最终报告优先使用 `report/figures/` 下的图片。

### `report/figures/pipeline_example.png`

完整流程示例图。  
展示原图、边缘检测、角点定位、透视矫正、阴影去除和最终二值图。  
适合放在报告的“系统流程与实验结果展示”部分。

### `report/figures/morphology_comparison.png`

形态学增强对比图。  
通常包含灰度图、阴影去除结果、形态学增强结果。  
适合说明形态学方法如何改善背景均匀性和文字对比度。

### `report/figures/binarization_comparison.png`

三种二值化方法对比图。  
包含固定阈值、Otsu、Sauvola 的结果。  
适合说明不同二值化方法在阴影和光照不均情况下的差异。

### `report/figures/canny_ablation.png`

Canny 参数消融图。  
展示不同 low/high 阈值下的边缘检测结果。  
适合说明边缘检测参数对文档角点定位的影响。

### `report/figures/sauvola_ablation.png`

Sauvola 参数消融图。  
展示不同窗口大小和 `k` 值下的二值化结果。  
适合说明局部阈值参数对文字保留和噪声的影响。

## 五、运行方法

安装依赖：

```bash
pip install -r requirements.txt
```

运行完整主流程：

```bash
python main.py --input_dir data/raw --output_dir outputs
```

运行参数消融实验：

```bash
python experiments/ablation.py --input_dir data/raw --output_dir outputs/ablation
```

生成报告可用图片：

```bash
python experiments/run_all.py
```

## 六、作为初学者应该先看什么

建议按下面顺序查看：

```text
1. outputs/comparisons/
2. outputs/04_contours/
3. outputs/05_warped/
4. outputs/07_enhanced/
5. outputs/08_binary_fixed/
6. outputs/09_binary_otsu/
7. outputs/10_binary_sauvola/
8. report/figures/
9. outputs/metrics.csv
```

最重要的是先看 `outputs/comparisons/`，它能一眼看到完整流程是否成功。

## 七、项目结论

固定阈值方法最简单，但对阴影和光照变化敏感；Otsu 方法在光照较均匀时表现较好；Sauvola 方法使用局部阈值，更适合存在阴影和光照不均的文档。形态学阴影去除和顶帽/底帽增强能够改善背景均匀性，提高文字区域与背景之间的对比度。

如果老师需要复现实验，只需要将图片放入 `data/raw/`，然后依次运行主流程、消融实验和报告图生成脚本。本项目不包含任何深度学习模型、OCR 模型或预训练模型。
