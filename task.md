下面这些可以直接作为 **Codex 任务指令** 逐条丢给 Codex。建议不要一次性全部丢进去，而是 **按任务 1、任务 2、任务 3……逐步执行**，这样出错更容易定位。

我按“最终可提交给老师”的标准，把任务拆成了完整项目流程。

---

## 总体项目目标

项目名称建议定为：

**基于形态学增强的文档图像智能扫描与矫正系统**

技术路线：

```text
手机拍摄文档图像
→ 灰度化与去噪
→ Canny边缘检测
→ 轮廓检测与多边形近似
→ 文档四角点定位
→ 透视变换矫正
→ 光照/阴影校正
→ 形态学增强
→ 固定阈值 / Otsu / Sauvola 二值化对比
→ 输出中间结果与最终扫描图
→ 生成实验统计结果
```

---

# 给 Codex 的任务 1：创建完整项目结构

```text
请帮我创建一个完整的传统图像处理课程项目，项目名称为 document_scanner_morphology。

项目主题是：基于形态学增强的文档图像智能扫描与矫正系统。

要求：
1. 使用 Python + OpenCV 实现。
2. 不允许使用任何深度学习框架或预训练模型。
3. 项目需要支持批量处理 data/raw/ 目录下的手机拍摄文档图片。
4. 自动输出所有中间结果图片和最终结果图片。
5. 创建清晰的项目目录结构。
6. 提供 requirements.txt。
7. 提供 README.md。
8. 提供可直接运行的 main.py。

请创建以下目录结构：

document_scanner_morphology/
├── data/
│   ├── raw/
│   └── samples/
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
│   └── comparisons/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   ├── preprocessing.py
│   ├── document_detection.py
│   ├── perspective.py
│   ├── morphology_enhancement.py
│   ├── binarization.py
│   ├── evaluation.py
│   └── visualization.py
├── experiments/
│   ├── run_all.py
│   └── ablation.py
├── report/
│   └── figures/
├── main.py
├── requirements.txt
└── README.md

先只完成项目结构、基础配置文件和空函数框架，不要急着写复杂算法。
```

---

# 给 Codex 的任务 2：实现图像读取、保存和批处理框架

```text
请在已有项目 document_scanner_morphology 中实现基础 I/O 和批处理框架。

要求：
1. 在 src/io_utils.py 中实现：
   - list_images(input_dir)：读取 jpg、jpeg、png、bmp 格式图片路径。
   - read_image(path)：使用 OpenCV 读取图片，失败时给出明确错误。
   - save_image(path, image)：自动创建目录并保存图片。
   - resize_for_processing(image, max_width=1200)：如果图像太大，按比例缩放，保留缩放比例。

2. 在 main.py 中实现批处理逻辑：
   - 默认输入目录为 data/raw/
   - 默认输出目录为 outputs/
   - 支持命令行参数：
     --input_dir
     --output_dir
     --max_width
     --debug
   - 对每张图片调用完整处理流程 process_document(image, name, output_dir)。

3. 在 src/config.py 中集中定义默认参数，例如：
   - MAX_WIDTH = 1200
   - GAUSSIAN_KERNEL = 5
   - CANNY_LOW = 50
   - CANNY_HIGH = 150
   - MORPH_KERNEL_SIZE = 15
   - FIXED_THRESHOLD = 127

4. 代码需要有清晰注释，方便我写课程报告。
```

---

# 给 Codex 的任务 3：实现预处理模块

```text
请实现 src/preprocessing.py 中的图像预处理模块。

要求实现以下函数：

1. to_gray(image)
   - 将 BGR 图像转为灰度图。

2. denoise_gray(gray, kernel_size=5)
   - 使用高斯滤波去噪。
   - kernel_size 必须为奇数。
   - 解释注释中说明高斯滤波用于抑制噪声，避免边缘检测产生伪边缘。

3. equalize_hist(gray)
   - 实现直方图均衡化。
   - 用于增强低对比度文档图像。

4. preprocess(image, config)
   - 返回 gray, blurred, equalized 三个结果。

注意：
1. 不要使用深度学习。
2. 只使用 OpenCV 和 numpy。
3. 每个函数都要写 docstring，说明输入、输出和算法作用。
```

---

# 给 Codex 的任务 4：实现文档边缘检测与轮廓检测

```text
请实现 src/document_detection.py，用于自动定位手机拍摄文档的四个角点。

要求实现以下函数：

1. detect_edges(gray_or_blur, low_threshold=50, high_threshold=150)
   - 使用 Canny 边缘检测。
   - 返回 edges 图像。

2. find_document_contour(edges, image_shape)
   - 使用 cv2.findContours 查找外部轮廓。
   - 按面积从大到小排序。
   - 使用 cv2.approxPolyDP 对轮廓做多边形近似。
   - 优先选择四边形轮廓作为文档边界。
   - 如果没有找到四边形，则返回面积最大的轮廓的最小外接矩形作为兜底方案。
   - 返回 4 个角点，格式为 numpy array，shape=(4, 2)。

3. draw_document_contour(image, corners)
   - 在原图上画出检测到的四边形边界和角点。
   - 返回可视化图像。

4. order_points(pts)
   - 将四个角点排序为：
     top-left, top-right, bottom-right, bottom-left。
   - 使用坐标和与差值方法实现。

注意：
1. 需要考虑检测失败的情况，并给出明确异常提示。
2. 代码要鲁棒，不要因为某张图片检测失败导致整个批处理停止。
3. 注释中说明 approxPolyDP 的 epsilon 参数含义。
```

---

# 给 Codex 的任务 5：实现透视变换矫正

```text
请实现 src/perspective.py，用于根据文档四角点进行透视矫正。

要求实现以下函数：

1. compute_output_size(ordered_pts)
   - 输入排序后的四个角点：
     top-left, top-right, bottom-right, bottom-left。
   - 根据上下边宽度、左右边高度计算输出图像宽高。
   - width = max(width_top, width_bottom)
   - height = max(height_left, height_right)

2. four_point_transform(image, pts)
   - 调用 order_points 对角点排序。
   - 计算目标矩形坐标。
   - 使用 cv2.getPerspectiveTransform 计算透视变换矩阵。
   - 使用 cv2.warpPerspective 得到矫正后的文档图像。
   - 返回 warped 图像和透视变换矩阵 M。

3. 在注释中说明透视变换的基本原理：
   - 将原图中的任意四边形区域映射为标准矩形区域。
   - 用于修正手机拍摄文档时产生的倾斜和透视畸变。

要求：
1. 不使用深度学习。
2. 代码需要考虑异常输入。
3. 保证输出图像方向正常。
```

---

# 给 Codex 的任务 6：实现形态学阴影去除与增强

```text
请实现 src/morphology_enhancement.py，用于文档图像的阴影去除和文字增强。

要求实现以下函数：

1. remove_shadow(gray, kernel_size=31)
   方法：
   - 使用较大的结构元素进行形态学闭运算，估计背景光照。
   - 用原灰度图除以背景图或做差分归一化，实现光照校正。
   - 返回 shadow_removed 图像。

2. top_hat_enhance(gray, kernel_size=15)
   - 使用形态学顶帽变换增强亮背景上的暗文字细节。
   - 返回增强结果。

3. black_hat_enhance(gray, kernel_size=15)
   - 使用形态学底帽变换突出暗文字或阴影区域。
   - 返回增强结果。

4. enhance_document(gray, kernel_size=15)
   - 综合使用阴影去除、顶帽/底帽变换、对比度归一化。
   - 返回 enhanced 图像。

5. create_kernel(kernel_size)
   - 创建矩形结构元素。
   - kernel_size 必须是正奇数。

注释要求：
1. 说明开运算、闭运算、顶帽、底帽的作用。
2. 说明 kernel_size 对结果的影响：
   - 太小：阴影估计不足；
   - 太大：可能损失局部文字细节；
   - 需要根据文档分辨率调整。
```

---

# 给 Codex 的任务 7：实现三种二值化方法对比

```text
请实现 src/binarization.py，用于比较不同二值化方法。

要求实现以下函数：

1. fixed_threshold(gray, threshold=127)
   - 固定阈值二值化。

2. otsu_threshold(gray)
   - Otsu 自适应全局阈值二值化。
   - 返回二值图和自动计算出的阈值。

3. sauvola_threshold(gray, window_size=25, k=0.2, R=128)
   - 手动实现 Sauvola 局部自适应阈值方法。
   - 不要直接调用 skimage 的 threshold_sauvola。
   - 公式：
     T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]
     其中 m 和 s 是局部窗口内的均值和标准差。
   - 可使用 cv2.boxFilter 高效计算局部均值和方差。
   - 返回二值图。

4. postprocess_binary(binary, kernel_size=3)
   - 使用形态学开运算去除小噪点。
   - 使用形态学闭运算连接断裂笔画。
   - 返回优化后的二值图。

5. compare_binarization(gray)
   - 同时返回 fixed、otsu、sauvola 三种结果。

注意：
1. 二值图要求文字为黑色，背景为白色。
2. 函数要写清楚 docstring。
3. 代码中要保留 Sauvola 公式注释，方便写报告。
```

---

# 给 Codex 的任务 8：串联完整文档扫描流程

```text
请把前面各模块串联起来，实现完整的文档扫描与增强流程。

要求：

1. 在 main.py 或 src/pipeline.py 中实现 process_document(image, image_name, output_dir, config)。

完整流程如下：
   原图
   → resize
   → gray
   → Gaussian blur
   → Canny edges
   → document contour detection
   → draw contour visualization
   → perspective transform
   → warped gray
   → shadow removal
   → morphology enhancement
   → fixed threshold
   → Otsu threshold
   → Sauvola threshold
   → binary post-processing
   → save all intermediate results

2. 每张图片至少保存以下结果：
   outputs/01_gray/xxx_gray.png
   outputs/02_blur/xxx_blur.png
   outputs/03_edges/xxx_edges.png
   outputs/04_contours/xxx_contour.png
   outputs/05_warped/xxx_warped.png
   outputs/06_shadow_removed/xxx_shadow_removed.png
   outputs/07_enhanced/xxx_enhanced.png
   outputs/08_binary_fixed/xxx_fixed.png
   outputs/09_binary_otsu/xxx_otsu.png
   outputs/10_binary_sauvola/xxx_sauvola.png
   outputs/comparisons/xxx_compare.png

3. 如果某张图片角点检测失败：
   - 不要让程序崩溃。
   - 记录失败信息。
   - 可以跳过该图像。
   - 在终端打印 warning。

4. comparison 图需要横向拼接展示：
   original / contour / warped / enhanced / fixed / otsu / sauvola

5. 代码需要能通过以下命令运行：
   python main.py --input_dir data/raw --output_dir outputs
```

---

# 给 Codex 的任务 9：实现实验评价指标

```text
请实现 src/evaluation.py，用于对文档扫描与增强结果进行简单定量评价。

因为该课程项目没有真实 OCR 标注，所以请实现以下无监督图像质量指标：

1. calculate_contrast(gray)
   - 返回图像灰度标准差，作为对比度指标。

2. calculate_sharpness(gray)
   - 使用 Laplacian 方差作为清晰度指标。

3. foreground_ratio(binary)
   - 计算二值图中黑色文字像素比例。
   - 用于分析二值化是否过度或不足。

4. edge_density(gray)
   - 使用 Canny 后计算边缘像素比例。

5. evaluate_image_results(image_name, enhanced, fixed, otsu, sauvola)
   - 计算增强图和三种二值化结果的指标。
   - 返回字典。

6. save_metrics_csv(metrics_list, output_csv)
   - 保存所有图片的评价结果到 outputs/metrics.csv。

评价指标字段包括：
   image_name
   enhanced_contrast
   enhanced_sharpness
   fixed_foreground_ratio
   otsu_foreground_ratio
   sauvola_foreground_ratio
   fixed_edge_density
   otsu_edge_density
   sauvola_edge_density

要求：
1. 指标不需要特别复杂，但要能用于课程报告中的定量分析。
2. 在 README 中解释每个指标的含义。
```

---

# 给 Codex 的任务 10：实现参数消融实验

```text
请实现 experiments/ablation.py，用于进行参数消融实验，方便写报告。

要求：

1. 对 Canny 参数进行实验：
   - low/high 分别取：
     (30, 100)
     (50, 150)
     (80, 200)

2. 对形态学 kernel_size 进行实验：
   - kernel_size 取：
     7
     15
     31

3. 对 Sauvola 参数进行实验：
   - window_size 取：
     15
     25
     35
   - k 取：
     0.2
     0.3
     0.5

4. 每组实验保存输出图像到：
   outputs/ablation/

5. 生成 ablation_results.csv，记录每组参数下的评价指标。

6. 生成可视化对比图：
   - Canny 不同阈值对边缘检测的影响
   - kernel_size 对阴影去除效果的影响
   - Sauvola 参数对二值化效果的影响

7. 运行方式：
   python experiments/ablation.py --input_dir data/raw --output_dir outputs/ablation

注意：
1. 只选择前 3 张图片做消融实验，避免输出太多。
2. 代码需要有注释，方便我在报告中分析参数影响。
```

---

# 给 Codex 的任务 11：生成可用于报告的结果图

```text
请实现 src/visualization.py 和 experiments/run_all.py，用于生成课程报告中可以直接使用的结果图。

要求：

1. src/visualization.py 实现：
   - put_title(image, title)
   - stack_images_horizontally(images, titles)
   - stack_images_grid(images, titles, cols=3)
   - save_comparison_figure(images, titles, output_path)

2. experiments/run_all.py 实现：
   - 批量处理 data/raw/ 所有图片。
   - 自动选择若干代表性图片生成报告图。
   - 将报告用图片保存到 report/figures/。

3. report/figures/ 至少生成：
   - pipeline_example.png
     包含：原图、边缘检测、角点定位、透视矫正、阴影去除、最终二值图。
   - binarization_comparison.png
     包含：固定阈值、Otsu、Sauvola 三种方法对比。
   - morphology_comparison.png
     包含：原灰度图、阴影去除、形态学增强图。
   - canny_ablation.png
     包含不同 Canny 参数结果。
   - sauvola_ablation.png
     包含不同 Sauvola 参数结果。

4. 所有图像标题使用英文即可，但 README 中说明中文含义。
```

---

# 给 Codex 的任务 12：完善 README 文档

```text
请为项目 document_scanner_morphology 写一个完整 README.md。

README 需要包括以下内容：

1. 项目简介
   - 说明本项目是传统图像处理课程项目。
   - 任务是实现手机拍摄文档图像的自动矫正、阴影去除和二值化增强。
   - 明确说明没有使用深度学习框架和预训练模型。

2. 项目结构说明。

3. 环境配置
   - Python 版本建议。
   - requirements.txt 安装方法：
     pip install -r requirements.txt

4. 数据准备
   - 将手机拍摄的文档图片放入 data/raw/。
   - 建议图片类型：
     A4纸、试卷、实验报告、书页、带阴影文档、倾斜文档等。

5. 运行方式
   - python main.py --input_dir data/raw --output_dir outputs
   - python experiments/ablation.py --input_dir data/raw --output_dir outputs/ablation
   - python experiments/run_all.py

6. 算法流程
   - 灰度化
   - 高斯滤波
   - Canny边缘检测
   - 轮廓检测
   - 多边形近似
   - 透视变换
   - 形态学阴影去除
   - 顶帽/底帽增强
   - 固定阈值、Otsu、Sauvola二值化

7. 实验输出说明
   - 每个 outputs 子目录保存什么结果。

8. 评价指标说明
   - contrast
   - sharpness
   - foreground_ratio
   - edge_density

9. 参数说明
   - Canny low/high threshold
   - morphology kernel_size
   - Sauvola window_size 和 k

10. 项目结论
   - Sauvola 更适合阴影和光照不均文档。
   - Otsu 在光照均匀时效果较好。
   - 固定阈值鲁棒性较差。
   - 形态学增强能改善文字对比度和背景均匀性。

README 要写得像一个可以提交给老师的课程项目说明。
```

---

# 给 Codex 的任务 13：生成课程报告初稿

```text
请在 report/ 目录下生成一份课程项目报告初稿，文件名为 report.md。

报告题目：
基于形态学增强的文档图像智能扫描与矫正系统

报告格式参考中文核心期刊或 IEEE 论文结构。

报告内容包括：

1. 题目与作者信息
   - 项目名称
   - 学生姓名：请留空，用“XXX”占位
   - 学号：请留空，用“XXX”占位
   - 完成日期：请留空，用“XXXX年XX月XX日”占位

2. 摘要
   - 约 300 字。
   - 包含研究背景、方法、实验、结论。

3. 关键词
   - 文档图像处理；透视矫正；形态学增强；二值化；OpenCV

4. 引言
   - 说明手机拍摄文档存在倾斜、阴影、光照不均等问题。
   - 说明传统图像处理方法的意义。
   - 说明本文工作内容。

5. 算法原理与实现
   包括：
   5.1 图像预处理
   5.2 Canny 边缘检测
   5.3 轮廓检测与多边形近似
   5.4 四点透视变换
   5.5 基于形态学的阴影去除
   5.6 顶帽和底帽变换
   5.7 固定阈值、Otsu 和 Sauvola 二值化
   5.8 算法流程伪代码

6. 实验设计
   - 自建手机拍摄文档数据集。
   - 图像数量先写 20 张，可后续修改。
   - 采集条件包括：正常光照、阴影、倾斜、复杂背景。
   - 对比方法包括：固定阈值、Otsu、Sauvola。
   - 消融实验包括：Canny 阈值、形态学核大小、Sauvola 参数。

7. 实验结果与分析
   - 预留插图位置：
     图1 系统流程图
     图2 文档角点检测结果
     图3 透视矫正结果
     图4 形态学增强前后对比
     图5 三种二值化方法对比
     图6 参数消融实验结果
   - 分析不同方法的优缺点。
   - 说明 Sauvola 对阴影场景更鲁棒。
   - 说明 Otsu 在光照均匀场景表现较好。
   - 说明固定阈值不适合复杂光照。

8. 讨论
   - 分析算法优点。
   - 分析局限性：
     文档边缘不完整时角点定位可能失败；
     强阴影下二值化仍可能丢失笔画；
     复杂背景可能干扰轮廓检测。
   - 改进方向：
     引入更鲁棒的角点排序；
     加入自适应参数选择；
     结合 OCR 评价识别准确率。

9. 结论与展望

10. 参考文献
   至少给出 8 篇经典文献或资料，包括：
   - Canny 边缘检测论文
   - Otsu 阈值论文
   - Sauvola 二值化论文
   - OpenCV 文档
   - 数字图像处理教材
   - 形态学图像处理相关文献

要求：
1. 报告中不要说使用了深度学习。
2. 报告内容要和项目代码一致。
3. 报告可以先写成 Markdown，后续我会转 Word。
4. 需要在报告中标注哪些地方需要我自己替换图片或实验数据。
```

---

# 给 Codex 的任务 14：检查代码是否能完整运行

```text
请对整个 document_scanner_morphology 项目进行一次完整检查和修复。

检查内容：

1. requirements.txt 是否完整。
2. main.py 是否可以运行。
3. experiments/ablation.py 是否可以运行。
4. experiments/run_all.py 是否可以运行。
5. data/raw/ 没有图片时，程序是否给出友好提示，而不是崩溃。
6. 某张图片文档角点检测失败时，是否跳过并继续处理其他图片。
7. outputs/ 是否能自动创建。
8. 所有保存图片的路径是否正确。
9. metrics.csv 是否能正确生成。
10. README 中的运行命令是否和实际代码一致。

请你直接修复发现的问题，并最后给出以下命令的运行说明：

python main.py --input_dir data/raw --output_dir outputs
python experiments/ablation.py --input_dir data/raw --output_dir outputs/ablation
python experiments/run_all.py
```

---

# 给 Codex 的任务 15：生成示例测试图片或说明

```text
请为项目添加一个 data/README.md，说明如何准备实验图片。

要求：
1. 不需要使用公开数据集，可以使用自建手机拍摄文档集。
2. 建议至少拍摄 20 张图片。
3. 图片类型包括：
   - 普通 A4 文档
   - 实验报告
   - 试卷
   - 书本页面
   - 有阴影的文档
   - 倾斜拍摄的文档
   - 背景复杂的文档
4. 说明采集条件：
   - 手机拍摄
   - 分辨率不限
   - 不同光照条件
   - 不同倾斜角度
5. 说明标注方式：
   - 本项目不需要像素级人工标注。
   - 可人工记录文档角点检测是否成功。
   - 可人工观察二值化结果并进行主观评价。
6. 说明命名方式：
   doc_001.jpg, doc_002.jpg, ...
```

---

# 给 Codex 的任务 16：最后整理成可提交版本

```text
请把项目整理成最终可提交版本。

要求：

1. 删除无用临时文件，例如：
   __pycache__
   .pytest_cache
   .DS_Store
   多余的测试输出

2. 保留以下内容：
   - src/
   - experiments/
   - main.py
   - requirements.txt
   - README.md
   - data/README.md
   - report/report.md
   - report/figures/
   - outputs/comparisons/ 中的代表性结果图
   - outputs/metrics.csv
   - outputs/ablation/ablation_results.csv

3. 新增 .gitignore，忽略：
   __pycache__/
   *.pyc
   .DS_Store
   .vscode/
   outputs/*
   !outputs/comparisons/
   !outputs/metrics.csv
   data/raw/*
   !data/raw/.gitkeep

4. 确认 README 中说明：
   - 如果老师需要复现实验，需要将图片放入 data/raw/ 后运行命令。
   - 项目不包含深度学习模型。
   - 所有核心算法基于传统图像处理方法。

5. 最后输出一个 FINAL_CHECKLIST.md，列出提交前检查清单。
```

---

## 你实际执行 Codex 的顺序

建议按这个顺序来：

```text
任务1：创建项目结构
任务2：实现 I/O 和批处理
任务3：实现预处理
任务4：实现角点检测
任务5：实现透视矫正
任务6：实现形态学增强
任务7：实现二值化方法
任务8：串联完整流程
任务9：实现评价指标
任务10：实现消融实验
任务11：生成报告用图
任务12：写 README
任务13：写课程报告初稿
任务14：完整运行检查
任务15：写数据集说明
任务16：整理提交版本
```

---

## 最关键的一条提醒

你让 Codex 写代码时，一定要反复强调：

**禁止使用深度学习框架或预训练模型，只能使用传统图像处理方法、OpenCV 基础函数和 NumPy。**

否则它可能会偷偷引入 OCR、深度学习文档检测、预训练模型之类的东西，这不符合课程要求。
