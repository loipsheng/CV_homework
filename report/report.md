# 基于形态学增强的文档图像智能扫描与矫正系统

学生姓名（学号）：劳业成 202534561015；王一评 202534261012  
完成日期：2026年6月15日

## 摘要

手机拍摄或扫描获得的文档图像常存在透视畸变、阴影、光照不均、模糊和背景干扰等退化问题，会降低文档可读性并影响后续文本识别与归档。针对该问题，本文设计并实现了一套基于传统图像处理的文档图像扫描与增强系统。系统首先对输入图像进行灰度化和高斯滤波，通过 Canny 边缘检测提取结构边缘，并结合外部轮廓检测与多边形近似自动定位文档四角；随后利用四点透视变换将倾斜文档校正为正视矩形图像；在增强阶段，采用形态学闭运算估计背景光照并进行阴影校正，再结合顶帽和底帽变换增强文字与背景的对比度；最后比较固定阈值、Otsu 和 Sauvola 三种二值化方法，并使用对比度、Laplacian 清晰度、前景像素比例和边缘密度等无监督指标进行评价。实验数据来源于 DocRes 文档图像复原任务相关公开样例，共选取 11 张文档图像进行全流程测试。实验结果表明，系统能够完成 11/11 张图像的批量处理；Sauvola 方法平均前景比例为 0.2725，高于固定阈值的 0.2356 和 Otsu 的 0.2467，在复杂光照下更容易保留局部文字细节；Canny 阈值提高会使边缘密度由 0.1345 降至 0.0443，说明阈值选择会显著影响文档边界定位。结果证明，基于形态学增强与局部阈值的传统方法能够有效改善文档图像的扫描质量。

关键词：文档图像处理；Canny 边缘检测；透视矫正；形态学增强；Sauvola 二值化；OpenCV

## 1 引言

随着手机拍摄成为常见的文档采集方式，文档图像经常受到拍摄角度、纸面弯曲、环境光照和背景杂物的影响。典型退化包括文档边缘倾斜、透视畸变、局部阴影、亮度不均和文字对比度下降。DocRes 论文指出，文档图像复原任务通常包括去弯曲、去阴影、外观增强、去模糊和二值化等多个方向，这些退化会影响文档分析系统的性能和人工阅读体验。

本项目不采用深度学习框架或预训练模型，而是基于 OpenCV 基础函数和 NumPy 实现一套可解释的传统图像处理流程。系统围绕课程要求完成三项任务：第一，实现基于 Canny 边缘检测与轮廓多边形近似的文档角点自动定位，并通过透视变换完成倾斜校正；第二，比较固定阈值、Otsu 和 Sauvola 三种二值化方法在阴影去除与文字增强后的表现；第三，在 11 张来源于 DocRes 相关公开样例的文档图像上进行全流程测试，并展示角点检测、形态学增强和最终二值图像等中间结果。

## 2 算法原理与实现

### 2.1 系统总体流程

系统输入为 `data/raw/` 目录中的文档图像，输出包括灰度图、滤波图、边缘图、角点检测图、透视矫正图、阴影去除图、形态学增强图、三种二值化图、横向对比图和定量指标表。整体流程如下：

```text
Input image
→ resize
→ grayscale
→ Gaussian blur
→ Canny edge detection
→ contour detection and polygon approximation
→ four-corner ordering
→ perspective transform
→ shadow removal
→ morphological enhancement
→ fixed / Otsu / Sauvola binarization
→ binary post-processing
→ save figures and metrics
```

### 2.2 图像预处理

输入图像首先由 BGR 彩色空间转换为灰度图。灰度化能够减少计算量，并使后续边缘检测、形态学操作和阈值分割都在单通道图像上进行。设输入彩色图像为 \(I(x,y)\)，灰度图记为 \(G(x,y)\)。OpenCV 的灰度化本质上是对 RGB 三通道进行加权组合。

为降低噪声对边缘检测的影响，系统对灰度图使用高斯滤波：

```text
B(x,y) = G(x,y) * K_sigma(x,y)
```

其中 \(K_\sigma\) 为高斯核，`*` 表示卷积。本项目默认高斯核大小为 5。核过小会保留较多噪声，核过大则可能削弱文档边缘，因此在实验中选择中等大小作为折中。

### 2.3 Canny 边缘检测

Canny 算法通过梯度计算、非极大值抑制和双阈值连接得到边缘图。梯度幅值可表示为：

```text
M(x,y) = sqrt(Gx(x,y)^2 + Gy(x,y)^2)
```

其中 \(G_x\) 和 \(G_y\) 分别为水平和垂直方向梯度。双阈值策略使用低阈值 \(T_l\) 和高阈值 \(T_h\)：高于 \(T_h\) 的像素被视为强边缘，介于两者之间且与强边缘连通的像素被保留为弱边缘。本项目默认参数为 `CANNY_LOW=50`、`CANNY_HIGH=150`。

消融实验表明，Canny 阈值从 `(30,100)` 提升到 `(80,200)` 时，前三张图像的平均边缘密度从 0.1345 降至 0.0443。说明低阈值能保留更多边缘，但也可能引入背景噪声；高阈值边缘更干净，但可能丢失文档边界。

### 2.4 轮廓检测与多边形近似

在 Canny 边缘图上使用 `cv2.findContours` 查找外部轮廓，并按轮廓面积从大到小排序。对候选轮廓使用 `cv2.approxPolyDP` 进行多边形近似：

```text
approx = approxPolyDP(contour, epsilon, closed=True)
epsilon = 0.02 × perimeter(contour)
```

其中 `epsilon` 表示原轮廓与近似多边形之间允许的最大距离。`epsilon` 越大，近似结果越简化；`epsilon` 越小，越接近原始轮廓但可能保留过多细节。本项目优先选取顶点数为 4 的轮廓作为文档区域。如果未找到四边形，则使用最大轮廓的最小外接矩形作为兜底策略。

四个角点通过坐标和与坐标差排序为左上、右上、右下、左下。该排序保证后续透视变换方向正确。

### 2.5 四点透视变换

文档角点排序后，系统计算目标矩形宽高：

```text
width = max(||TR - TL||, ||BR - BL||)
height = max(||BL - TL||, ||BR - TR||)
```

设原图四点为 \(p_i=(x_i,y_i)\)，目标矩形四点为 \(q_i=(u_i,v_i)\)。透视变换通过单应矩阵 \(H\) 将原图坐标映射到目标坐标：

```text
s [u, v, 1]^T = H [x, y, 1]^T
```

其中 \(s\) 为尺度因子。程序使用 `cv2.getPerspectiveTransform` 求解矩阵 \(H\)，再用 `cv2.warpPerspective` 得到校正图像。该步骤用于消除手机斜拍产生的透视畸变。

### 2.6 形态学阴影去除

文档中的阴影通常表现为缓慢变化的背景亮度。系统使用较大的矩形结构元素进行形态学闭运算估计背景：

```text
Background = close(G, B) = erode(dilate(G, B), B)
```

其中 \(B\) 为结构元素。闭运算能够填补较暗的小区域，得到较平滑的背景估计。之后将原灰度图与背景图相除并归一化：

```text
S(x,y) = normalize(255 × G(x,y) / Background(x,y))
```

得到阴影校正图。默认阴影估计核大小为 31。核太小会导致背景估计不足，核太大可能损失局部文字细节。

### 2.7 顶帽、底帽与形态学增强

顶帽变换定义为原图与开运算结果之差：

```text
TopHat(G) = G - open(G, B)
```

底帽变换定义为闭运算结果与原图之差：

```text
BlackHat(G) = close(G, B) - G
```

顶帽可突出亮背景上的局部细节，底帽可突出暗文字和暗区域。系统将阴影去除图、顶帽结果和底帽结果组合：

```text
Enhanced = normalize(ShadowRemoved + TopHat - BlackHat)
```

消融实验显示，形态学核从 7 增大到 31 时，前三张图像的平均对比度从 74.24 提高到 80.70，但清晰度从 17988.96 下降到 13802.32，边缘密度从 0.2666 降到 0.2388。说明较大核有利于背景均匀化和对比度提升，但可能平滑部分文字细节。

### 2.8 固定阈值、Otsu 与 Sauvola 二值化

固定阈值方法使用统一阈值 \(T=127\)：

```text
Binary(x,y) = 255, if G(x,y) > T
Binary(x,y) = 0, otherwise
```

该方法简单，但对光照变化敏感。

Otsu 方法通过最大化类间方差自动寻找全局阈值。设前景和背景两类权重为 \(\omega_0,\omega_1\)，均值为 \(\mu_0,\mu_1\)，则类间方差为：

```text
sigma_b^2(t) = omega_0(t) × omega_1(t) × [mu_0(t) - mu_1(t)]^2
```

使 \(\sigma_b^2(t)\) 最大的 \(t\) 即为最佳阈值。Otsu 适合前景与背景灰度分布较明显、光照较均匀的图像。

Sauvola 方法是局部自适应阈值方法，其阈值为：

```text
T(x,y) = m(x,y) × [1 + k × (s(x,y) / R - 1)]
```

其中 \(m(x,y)\) 和 \(s(x,y)\) 分别为局部窗口内的均值和标准差，\(k\) 为调节系数，\(R\) 为标准差动态范围。本项目默认 `window_size=25`、`k=0.2`、`R=128`，并使用 `cv2.boxFilter` 高效计算局部均值和方差。

### 2.9 关键伪代码

```text
for each image in data/raw:
    image = read_image(path)
    resized = resize_for_processing(image)
    gray = to_gray(resized)
    blurred = gaussian_blur(gray)
    edges = canny(blurred, low=50, high=150)
    corners = find_document_contour(edges)
    contour_vis = draw_document_contour(image, corners)
    warped = four_point_transform(image, corners)
    warped_gray = to_gray(warped)
    shadow_removed = remove_shadow(warped_gray, kernel=31)
    enhanced = enhance_document(shadow_removed, kernel=15)
    fixed = fixed_threshold(enhanced, threshold=127)
    otsu = otsu_threshold(enhanced)
    sauvola = sauvola_threshold(enhanced, window=25, k=0.2)
    save all intermediate results
    calculate metrics
```

## 3 实验设计

### 3.1 数据集来源与图像数量

实验数据来源于论文 “DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks” 相关公开文档图像复原样例。DocRes 论文将文档图像复原归纳为去弯曲、去阴影、外观增强、去模糊和二值化等任务，本文选取其中具有代表性的 11 张文档图像作为实验输入。原始图片已统一重命名为 `doc_001` 至 `doc_011`，格式包括 PNG 和 JPG。

本文只使用这些图像作为传统图像处理实验数据，不使用 DocRes 模型、深度学习框架或任何预训练权重。

### 3.2 标注方式

本项目没有像素级人工标注。角点检测结果通过 `outputs/04_contours/` 中的可视化图进行人工观察：若绿色文档边框基本贴合纸张边界，则认为角点定位成功。二值化结果通过 `outputs/08_binary_fixed/`、`outputs/09_binary_otsu/`、`outputs/10_binary_sauvola/` 和 `outputs/comparisons/` 进行主观观察，同时结合前景比例和边缘密度进行定量分析。

### 3.3 评价指标

由于实验没有 OCR 标注和真实角点标注，本文采用无监督图像质量指标：

1. 对比度 `enhanced_contrast`：增强图像灰度标准差，反映文字与背景灰度差异。
2. 清晰度 `enhanced_sharpness`：Laplacian 方差，反映图像边缘与细节清晰程度。
3. 前景比例 `foreground_ratio`：二值图中黑色像素比例，用于判断文字保留是否过少或过多。
4. 边缘密度 `edge_density`：Canny 边缘像素比例，用于反映二值图保留的结构细节。

这些指标不代表绝对主观质量，但能支持不同方法之间的对比分析。

### 3.4 对比实验与消融实验

二值化对比实验包含三种基准方法：固定阈值、Otsu 和 Sauvola。它们分别代表人工固定全局阈值、自动全局阈值和局部自适应阈值。

参数消融实验包含三组：

1. Canny 阈值：`(30,100)`、`(50,150)`、`(80,200)`。
2. 形态学核大小：`7`、`15`、`31`。
3. Sauvola 参数：窗口大小 `15`、`25`、`35`，参数 `k=0.2`、`0.3`、`0.5`。

## 4 实验结果与分析

### 4.1 全流程处理结果

主程序对 11 张图像全部完成处理，成功率为 11/11。每张图像均保存灰度图、滤波图、边缘图、角点检测图、透视矫正图、阴影去除图、形态学增强图、三种二值化图和横向对比图。

建议在报告中插入：`report/figures/pipeline_example.png`。该图展示从原图到边缘检测、角点定位、透视矫正、阴影去除和最终二值化的完整流程。

### 4.2 角点检测与透视矫正分析

角点检测结果保存在 `outputs/04_contours/`。绿色轮廓框和红色角点用于验证文档边界定位是否正确。透视矫正结果保存在 `outputs/05_warped/`，用于观察倾斜文档是否被拉正为矩形区域。

实验中 Canny 阈值对边缘图质量影响明显。消融实验前三张图的平均边缘密度如下：

| Canny 参数 | 平均边缘密度 |
|---|---:|
| low=30, high=100 | 0.1345 |
| low=50, high=150 | 0.0972 |
| low=80, high=200 | 0.0443 |

低阈值产生更多边缘，可能有利于保留文档边界，但也更容易引入背景纹理；高阈值抑制噪声能力更强，但可能导致文档边缘不连续。因此默认选择 `(50,150)` 作为折中。

### 4.3 形态学增强效果分析

形态学增强结果保存在 `outputs/07_enhanced/`，报告图为 `report/figures/morphology_comparison.png`。从指标看，11 张图像增强后的平均对比度为 71.5487，平均清晰度为 7948.4415。消融实验中，形态学核大小对增强结果影响如下：

| kernel_size | 平均对比度 | 平均清晰度 | 平均边缘密度 |
|---:|---:|---:|---:|
| 7 | 74.2415 | 17988.9637 | 0.2666 |
| 15 | 76.8092 | 15573.1928 | 0.2474 |
| 31 | 80.6975 | 13802.3191 | 0.2388 |

可以看出，较大的核有助于提高背景均匀性和整体对比度，但也会降低清晰度和边缘密度，说明局部文字细节可能被平滑。综合考虑，主流程选择 `kernel_size=15`，在阴影估计和文字细节保留之间取得折中。

### 4.4 三种二值化方法对比

报告图 `report/figures/binarization_comparison.png` 展示了固定阈值、Otsu 和 Sauvola 的视觉对比。11 张图像的平均指标如下：

| 方法 | 平均前景比例 | 平均边缘密度 |
|---|---:|---:|
| 固定阈值 | 0.2356 | 0.0587 |
| Otsu | 0.2467 | 0.0623 |
| Sauvola | 0.2725 | 0.0690 |

固定阈值方法实现最简单，但无法适应局部光照变化；Otsu 自动选择全局阈值，在前景与背景灰度分布较清晰时表现较好；Sauvola 的平均前景比例和边缘密度最高，说明其能保留更多局部文字结构，更适合阴影和光照不均的文档。但当前景比例过高时也可能引入额外噪声，因此需要结合视觉结果判断。

### 4.5 Sauvola 参数消融分析

前三张图像的 Sauvola 消融实验表明，`k` 值增大会降低前景比例和边缘密度。例如在 `window_size=25` 时，平均前景比例由 `k=0.2` 的 0.2806 降至 `k=0.5` 的 0.2312，平均边缘密度由 0.0884 降至 0.0797。这说明较大的 `k` 会提高阈值抑制效果，使黑色前景减少，噪声可能下降，但弱文字也可能被丢失。

窗口大小从 15 增至 35 时，结果变化相对平缓。较小窗口更关注局部变化，较大窗口对背景变化更平滑。主流程选择 `window_size=25`、`k=0.2`，用于优先保留弱文字和阴影区域中的细节。

### 4.6 异常样本分析

在 `metrics.csv` 中，`doc_007.png` 的三种二值化前景比例和边缘密度均为 0，说明该图像在二值化阶段可能被处理为近似全白图。这类样本提示传统阈值方法对极端输入较敏感，可能需要针对低对比度、特殊背景或过度增强图像进行额外参数调整。

## 5 讨论

本系统的优点在于流程完整、结构清晰、可解释性强，能够输出每一步中间结果，便于课程报告分析。与端到端深度模型相比，传统方法不需要训练数据和 GPU，运行逻辑透明，适合教学场景。

系统局限主要包括三点。第一，角点定位依赖文档边缘，如果边缘被遮挡、背景轮廓干扰严重或纸张与背景颜色接近，轮廓检测可能失败。第二，形态学核大小需要根据图像分辨率调整，固定参数难以适应所有文档。第三，二值化指标缺乏 OCR 识别率或真实前景标注，因此当前定量分析只能反映图像统计特征，不能完全代表文字识别效果。

未来改进方向包括：引入更鲁棒的文档区域筛选策略；根据图像尺寸自适应选择形态学核大小；在有标注条件下加入角点误差、OCR 识别率或字符级准确率作为评价指标；针对异常样本设计自动参数回退机制。

## 6 结论与展望

本文完成了一套基于传统图像处理的文档图像智能扫描与矫正系统。系统实现了 Canny 边缘检测、轮廓多边形近似、四点透视变换、形态学阴影去除、顶帽/底帽增强以及三种二值化方法对比。实验数据来源于 DocRes 文档图像复原相关公开样例，共 11 张图像，主流程全部处理成功。

实验结果表明，Canny 阈值会显著影响边缘密度和角点检测基础；形态学核大小越大，背景均匀化越明显，但可能牺牲文字细节；Sauvola 局部阈值在平均前景比例和边缘密度上高于固定阈值与 Otsu，更适合阴影和光照不均场景。总体而言，传统 OpenCV 与 NumPy 方法能够实现可解释、可复现的文档扫描流程，但在复杂背景和极端退化图像上仍需进一步提升鲁棒性。

## 参考文献

[1] Jiaxin Zhang, Dezhi Peng, Chongyu Liu, Peirong Zhang, Lianwen Jin. DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks. arXiv:2405.04408, 2024.  
[2] J. Canny. A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6):679-698, 1986.  
[3] N. Otsu. A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62-66, 1979.  
[4] J. Sauvola and M. Pietikainen. Adaptive Document Image Binarization. Pattern Recognition, 33(2):225-236, 2000.  
[5] R. C. Gonzalez and R. E. Woods. Digital Image Processing. Prentice Hall, 3rd edition, 2008.  
[6] J. Serra. Image Analysis and Mathematical Morphology. Academic Press, 1982.  
[7] OpenCV Documentation. Image Processing in OpenCV: Canny Edge Detection, Morphological Transformations, Thresholding and Geometric Transformations.  
[8] G. Bradski. The OpenCV Library. Dr. Dobb's Journal of Software Tools, 2000.
