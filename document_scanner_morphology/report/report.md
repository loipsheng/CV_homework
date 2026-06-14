# 基于形态学增强的文档图像智能扫描与矫正系统

学生姓名：XXX  
学号：XXX  
完成日期：XXXX年XX月XX日

## 摘要

手机拍摄文档图像常受到倾斜、透视畸变、阴影和光照不均影响，导致文档阅读质量下降，并影响后续文字识别或人工归档。本文设计并实现一种基于传统图像处理的文档扫描与增强系统。系统首先对输入图像进行灰度化和高斯滤波，通过 Canny 边缘检测与轮廓多边形近似自动定位文档四角，再利用四点透视变换完成倾斜矫正。随后，采用形态学闭运算估计背景光照并进行阴影校正，结合顶帽和底帽变换增强文字细节。最后，对固定阈值、Otsu 和 Sauvola 三种二值化方法进行对比，并使用对比度、清晰度、前景比例和边缘密度等无监督指标进行分析。实验结果表明，形态学增强能够改善文档背景均匀性，Sauvola 局部阈值在阴影和光照不均场景中更稳定，Otsu 在光照均匀图像上具有较好效果。

关键词：文档图像处理；透视矫正；形态学增强；二值化；OpenCV

## 1 引言

手机拍摄文档常见问题包括拍摄角度倾斜、透视畸变、局部阴影、纸面反光和复杂背景干扰。这些问题会降低文档可读性，也会影响后续 OCR 或人工阅览。传统图像处理方法具有可解释性强、实现简单、无需训练数据等特点，适合课程项目中对底层视觉算法进行理解和实践。

本文实现的系统包括文档角点检测、透视矫正、形态学阴影去除、文字增强和二值化对比实验。所有算法均使用 OpenCV 基础函数和 NumPy 实现，不使用深度学习框架或预训练模型。

## 2 算法原理与实现

### 2.1 图像预处理

输入图像首先由 BGR 转换为灰度图。为了减少图像噪声对边缘检测的影响，使用高斯滤波进行平滑。高斯核大小默认为 5，核越大平滑越强，但也可能削弱边缘。

### 2.2 Canny 边缘检测

Canny 算法通过梯度计算、非极大值抑制和双阈值连接提取边缘。本文默认低阈值为 50，高阈值为 150。阈值较低时边缘更丰富，但噪声也更多；阈值较高时边缘更干净，但可能丢失文档边界。

### 2.3 轮廓检测与多边形近似

使用 `cv2.findContours` 检测外部轮廓，并按面积从大到小排序。对候选轮廓使用 `cv2.approxPolyDP` 进行多边形近似，其中 epsilon 表示原始轮廓到近似多边形的最大允许距离。若近似结果为四边形，则认为其为文档边界；若未找到四边形，则使用最大轮廓的最小外接矩形作为兜底。

### 2.4 四点透视变换

将检测到的四个角点排序为左上、右上、右下、左下，然后根据上下边宽度和左右边高度计算输出矩形尺寸。使用 `cv2.getPerspectiveTransform` 计算变换矩阵，并通过 `cv2.warpPerspective` 将原图中的任意四边形文档映射为标准矩形区域。

### 2.5 基于形态学的阴影去除

形态学闭运算可用于估计缓慢变化的背景光照。本文使用较大的矩形结构元素对灰度图进行闭运算，得到背景图，再用原图除以背景图并归一化，从而削弱阴影和光照不均。

### 2.6 顶帽和底帽变换

顶帽变换提取原图与开运算结果之间的差异，可突出亮背景上的局部细节。底帽变换提取闭运算结果与原图之间的差异，可突出暗文字或阴影区域。本文将阴影去除、顶帽增强和底帽抑制结合，用于提升文字对比度。

### 2.7 固定阈值、Otsu 和 Sauvola 二值化

固定阈值方法使用统一阈值 127，适合光照稳定图像。Otsu 方法自动选择全局阈值，适合前景和背景灰度分布较明显的图像。Sauvola 方法使用局部均值和标准差计算阈值：

```text
T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]
```

其中 `m` 为局部均值，`s` 为局部标准差，`k` 控制局部对比度影响，`R` 为标准差动态范围。本文使用 `cv2.boxFilter` 高效计算局部统计量。

### 2.8 算法流程伪代码

```text
for image in data/raw:
    resize image
    gray = convert_to_gray(image)
    blur = gaussian_blur(gray)
    edges = canny(blur)
    corners = find_document_contour(edges)
    warped = perspective_transform(image, corners)
    warped_gray = convert_to_gray(warped)
    shadow_removed = morphology_shadow_removal(warped_gray)
    enhanced = top_hat_black_hat_enhancement(shadow_removed)
    fixed = fixed_threshold(enhanced)
    otsu = otsu_threshold(enhanced)
    sauvola = sauvola_threshold(enhanced)
    save intermediate results and metrics
```

## 3 实验设计

数据集采用自建手机拍摄文档集，初步计划采集 20 张图像，包含正常光照、阴影、倾斜和复杂背景等条件。对比方法包括固定阈值、Otsu 和 Sauvola 二值化。消融实验包括 Canny 阈值、形态学核大小和 Sauvola 参数。

需要替换：请在最终提交前将实际图像数量、采集设备和采集场景补充到本节。

## 4 实验结果与分析

需要插入图 1：系统流程图，建议使用 `report/figures/pipeline_example.png`。  
需要插入图 2：文档角点检测结果，来源于 `outputs/04_contours/`。  
需要插入图 3：透视矫正结果，来源于 `outputs/05_warped/`。  
需要插入图 4：形态学增强前后对比，建议使用 `report/figures/morphology_comparison.png`。  
需要插入图 5：三种二值化方法对比，建议使用 `report/figures/binarization_comparison.png`。  
需要插入图 6：参数消融实验结果，建议使用 `report/figures/canny_ablation.png` 和 `report/figures/sauvola_ablation.png`。

从方法特点看，固定阈值实现简单，但无法适应复杂光照；Otsu 可以自动选择全局阈值，在光照较均匀场景中效果较好；Sauvola 使用局部统计量，因此在阴影和非均匀光照中更鲁棒。形态学增强能改善背景均匀性，使文字和纸面背景之间的对比更明显。

需要替换：请根据 `outputs/metrics.csv` 和 `outputs/ablation/ablation_results.csv` 补充实际数值表格。

## 5 讨论

本系统的优点是流程完整、可解释性强，不依赖训练数据，适合小规模课程实验。局限性在于当文档边缘不完整、背景轮廓干扰强或阴影极端严重时，角点定位可能失败；强阴影下二值化仍可能出现笔画断裂；复杂背景可能影响轮廓检测。后续可改进角点排序和边界筛选策略，加入自适应参数选择，并结合 OCR 识别率进行更直接的可读性评价。

## 6 结论与展望

本文完成了基于传统图像处理的文档扫描与矫正系统，实现了从手机拍摄图像到透视矫正、阴影去除、文字增强和二值化输出的完整流程。实验设计表明，形态学增强和 Sauvola 局部阈值对复杂光照文档具有较好适应性。未来可进一步提升复杂背景下的边界检测鲁棒性，并扩展更多客观评价指标。

## 参考文献

[1] J. Canny, "A Computational Approach to Edge Detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986.  
[2] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," IEEE Transactions on Systems, Man, and Cybernetics, 1979.  
[3] J. Sauvola and M. Pietikäinen, "Adaptive Document Image Binarization," Pattern Recognition, 2000.  
[4] R. C. Gonzalez and R. E. Woods, Digital Image Processing, Pearson.  
[5] OpenCV Documentation, Image Processing Module.  
[6] J. Serra, Image Analysis and Mathematical Morphology, Academic Press.  
[7] R. Szeliski, Computer Vision: Algorithms and Applications, Springer.  
[8] W. K. Pratt, Digital Image Processing, Wiley.

