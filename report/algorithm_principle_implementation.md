# 算法原理与实现补充材料

本文档只针对报告中“算法原理与实现”部分进行补充，可直接复制到正式报告对应章节中。内容包括分模块算法原理、公式、关键代码段/伪代码、参数选取依据和调试过程。

## 1. 总体处理流程

本项目实现的是一个传统图像处理文档扫描流程，输入为手机拍摄或扫描得到的文档图像，输出为透视矫正后的增强文档图和三种二值化结果。整体流程如下：

```text
输入文档图像
→ 图像缩放
→ 灰度化
→ 高斯滤波去噪
→ Canny 边缘检测
→ 外部轮廓检测
→ 多边形近似定位文档四角
→ 四点透视变换
→ 形态学阴影去除
→ 顶帽/底帽增强
→ 固定阈值、Otsu、Sauvola 二值化
→ 二值图形态学后处理
→ 保存中间结果、最终结果和评价指标
```

主流程伪代码如下：

```text
for each image in data/raw:
    image = read_image(path)
    resized = resize_for_processing(image, max_width=1200)
    gray = to_gray(resized)
    blurred = gaussian_blur(gray, kernel_size=5)
    edges = canny(blurred, low=50, high=150)
    corners = find_document_contour(edges)
    warped = four_point_transform(resized, corners)
    warped_gray = to_gray(warped)
    shadow_removed = remove_shadow(warped_gray, kernel_size=31)
    enhanced = enhance_document(shadow_removed, kernel_size=15)
    fixed = fixed_threshold(enhanced, threshold=127)
    otsu = otsu_threshold(enhanced)
    sauvola = sauvola_threshold(enhanced, window_size=25, k=0.2)
    fixed, otsu, sauvola = postprocess_binary(...)
    save all intermediate images and metrics
```

对应代码位置：`src/pipeline.py` 中的 `process_document()`。

核心代码段：

```python
resized, _ = resize_for_processing(image, config.MAX_WIDTH)
gray, blurred, _ = preprocess(resized, config)
edges = detect_edges(blurred, config.CANNY_LOW, config.CANNY_HIGH)
corners = find_document_contour(edges, resized.shape)
warped, _ = four_point_transform(resized, corners)
shadow_removed = remove_shadow(warped_gray, config.SHADOW_KERNEL_SIZE)
enhanced = enhance_document(shadow_removed, config.MORPH_KERNEL_SIZE)
fixed = fixed_threshold(enhanced, config.FIXED_THRESHOLD)
otsu, otsu_value = otsu_threshold(enhanced)
sauvola = sauvola_threshold(enhanced, config.SAUVOLA_WINDOW_SIZE, config.SAUVOLA_K)
```

该流程的设计思路是：先解决“文档在哪里”和“文档是否倾斜”的几何问题，再解决“阴影、光照和文字是否清楚”的增强问题，最后比较不同二值化方法对文字前景区域的分割效果。

## 2. 图像预处理模块

### 2.1 灰度化

输入图像通常是 BGR 彩色图像。为了减少计算量并统一后续处理，本项目首先将其转换为单通道灰度图。OpenCV 灰度化可理解为对 RGB 三个通道进行加权求和：

```text
G(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114B(x,y)
```

其中 \(G(x,y)\) 表示灰度图在像素位置 \((x,y)\) 的亮度值。

关键代码：

```python
def to_gray(image):
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 2.2 高斯滤波

文档图像中可能存在拍摄噪声、纸面纹理和压缩噪声。若直接进行边缘检测，容易产生大量伪边缘。因此在 Canny 之前使用高斯滤波进行平滑：

```text
B(x,y) = G(x,y) * K_sigma(x,y)
```

其中 \(K_\sigma\) 为高斯核，`*` 表示卷积。高斯核越大，平滑效果越强，但边缘也会被削弱。

关键代码：

```python
def denoise_gray(gray, kernel_size=5):
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
```

参数选取：

- `GAUSSIAN_KERNEL = 5`
- 选择原因：核大小 5 能抑制大部分小噪声，同时不会明显破坏文档边缘。
- 调试方式：观察 `outputs/02_blur/` 和 `outputs/03_edges/`。如果边缘图噪声过多，可增大核；如果文档边界变模糊，应减小核。

## 3. Canny 边缘检测模块

Canny 边缘检测用于提取文档边界。其基本步骤包括梯度计算、非极大值抑制和双阈值连接。

梯度幅值公式为：

```text
M(x,y) = sqrt(Gx(x,y)^2 + Gy(x,y)^2)
```

其中 \(G_x\) 和 \(G_y\) 分别表示水平和垂直方向梯度。Canny 使用两个阈值：

- 高阈值 \(T_h\)：高于该值的像素认为是强边缘。
- 低阈值 \(T_l\)：介于低阈值和高阈值之间，且与强边缘相连的像素作为弱边缘保留。

关键代码：

```python
def detect_edges(gray_or_blur, low_threshold=50, high_threshold=150):
    return cv2.Canny(gray_or_blur, low_threshold, high_threshold)
```

参数选取：

- `CANNY_LOW = 50`
- `CANNY_HIGH = 150`

选取依据：

- 阈值太低：边缘数量多，但背景纹理和噪声也会进入轮廓检测。
- 阈值太高：边缘更干净，但文档边界可能断裂。

调试结果：

消融实验中，前三张图像的平均边缘密度如下：

| Canny 参数 | 平均边缘密度 |
|---|---:|
| low=30, high=100 | 0.1345 |
| low=50, high=150 | 0.0972 |
| low=80, high=200 | 0.0443 |

可以看到，阈值升高后边缘密度明显下降。因此主流程选择 `(50,150)`，在边缘保留和噪声抑制之间折中。

## 4. 轮廓检测与文档角点定位

Canny 结果只是一张边缘图，还不能直接得到文档四个角。本项目使用 `cv2.findContours` 查找外部轮廓，并将轮廓按面积从大到小排序。一般情况下，文档区域是图像中面积较大的闭合轮廓。

多边形近似使用 Douglas-Peucker 思想，OpenCV 中对应函数为 `cv2.approxPolyDP`：

```text
approx = approxPolyDP(contour, epsilon, closed=True)
epsilon = 0.02 × perimeter(contour)
```

其中 `epsilon` 表示原始轮廓到近似多边形的最大允许距离：

- `epsilon` 较小：近似轮廓更接近原轮廓，顶点可能过多。
- `epsilon` 较大：轮廓更简化，但可能丢失真实角点。

关键伪代码：

```text
contours = findContours(edges)
sort contours by area descending
for contour in largest contours:
    perimeter = arcLength(contour)
    approx = approxPolyDP(contour, 0.02 * perimeter)
    if len(approx) == 4:
        return order_points(approx)
if no quadrilateral is found:
    return minAreaRect(largest_contour)
```

角点排序方法：

```text
top-left     = point with smallest x + y
bottom-right = point with largest  x + y
top-right    = point with smallest x - y
bottom-left  = point with largest  x - y
```

调试方式：

- 查看 `outputs/04_contours/`。
- 如果绿色框基本贴合文档边界，说明角点检测成功。
- 如果框到背景区域，通常说明 Canny 阈值、背景干扰或文档边缘不完整影响了轮廓检测。

## 5. 四点透视变换模块

手机拍摄文档常出现倾斜和透视畸变。检测到四个文档角点后，可通过透视变换将任意四边形区域映射为标准矩形。

设原图角点为：

```text
TL, TR, BR, BL
```

输出图像宽高计算为：

```text
width  = max(||TR - TL||, ||BR - BL||)
height = max(||BL - TL||, ||BR - TR||)
```

透视变换可表示为单应矩阵：

```text
s [u, v, 1]^T = H [x, y, 1]^T
```

其中：

- \((x,y)\)：原图坐标。
- \((u,v)\)：目标矩形坐标。
- \(H\)：3×3 透视变换矩阵。
- \(s\)：尺度因子。

关键代码：

```python
rect = order_points(pts)
width, height = compute_output_size(rect)
dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1],
], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(image, matrix, (width, height))
```

调试方式：

- 查看 `outputs/05_warped/`。
- 如果文档被拉成正视矩形，说明角点排序和透视变换正确。
- 如果图像倒置、扭曲或裁剪错误，通常是角点顺序或轮廓定位出现问题。

## 6. 形态学阴影去除模块

文档图像中的阴影可以看作缓慢变化的背景光照。为了削弱阴影，本项目使用形态学闭运算估计背景，再用原图除以背景图进行归一化。

闭运算定义为：

```text
close(G, B) = erode(dilate(G, B), B)
```

其中：

- \(G\)：输入灰度图。
- \(B\)：结构元素。
- `dilate`：膨胀。
- `erode`：腐蚀。

背景估计与阴影校正公式：

```text
Background(x,y) = close(G, B)
S(x,y) = normalize(255 × G(x,y) / Background(x,y))
```

关键代码：

```python
def remove_shadow(gray, kernel_size=31):
    kernel = create_kernel(kernel_size)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    corrected = cv2.divide(gray, background, scale=255)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
```

参数选取：

- `SHADOW_KERNEL_SIZE = 31`
- 选择原因：阴影通常是大范围缓慢变化区域，需要较大的结构元素估计背景。
- 调试方式：查看 `outputs/06_shadow_removed/`。

参数影响：

- 核太小：无法覆盖大范围阴影，背景估计不足。
- 核太大：可能把局部文字细节也当成背景变化抹掉。

## 7. 顶帽、底帽与文档增强模块

阴影去除后，文字仍可能较浅，因此进一步使用顶帽和底帽变换增强文字与背景差异。

开运算：

```text
open(G, B) = dilate(erode(G, B), B)
```

顶帽变换：

```text
TopHat(G) = G - open(G, B)
```

底帽变换：

```text
BlackHat(G) = close(G, B) - G
```

增强公式：

```text
Enhanced = normalize(ShadowRemoved + TopHat - BlackHat)
```

关键代码：

```python
shadow_removed = remove_shadow(gray, max(31, kernel_size * 2 + 1))
top_hat = top_hat_enhance(shadow_removed, kernel_size)
black_hat = black_hat_enhance(shadow_removed, kernel_size)
enhanced = cv2.add(shadow_removed, top_hat)
enhanced = cv2.subtract(enhanced, black_hat)
enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
```

参数选取：

- `MORPH_KERNEL_SIZE = 15`

消融实验结果：

| kernel_size | 平均对比度 | 平均清晰度 | 平均边缘密度 |
|---:|---:|---:|---:|
| 7 | 74.2415 | 17988.9637 | 0.2666 |
| 15 | 76.8092 | 15573.1928 | 0.2474 |
| 31 | 80.6975 | 13802.3191 | 0.2388 |

分析：

- 核增大后，对比度提升，说明背景均匀化更明显。
- 但清晰度和边缘密度下降，说明部分文字细节可能被平滑。
- 因此主流程选择 `15`，兼顾阴影去除和文字细节保留。

## 8. 二值化模块

二值化的目标是将增强后的灰度文档转成黑白图，黑色表示文字前景，白色表示背景。

### 8.1 固定阈值

固定阈值使用一个常数阈值 \(T=127\)：

```text
Binary(x,y) = 255, if G(x,y) > T
Binary(x,y) = 0, otherwise
```

关键代码：

```python
def fixed_threshold(gray, threshold=127):
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary
```

特点：

- 优点：简单、速度快。
- 缺点：无法适应阴影和光照不均。

### 8.2 Otsu 阈值

Otsu 方法自动寻找一个全局阈值，使前景和背景之间的类间方差最大。设两类权重为 \(\omega_0, \omega_1\)，均值为 \(\mu_0, \mu_1\)，类间方差为：

```text
sigma_b^2(t) = omega_0(t) × omega_1(t) × [mu_0(t) - mu_1(t)]^2
```

最佳阈值为：

```text
T* = argmax sigma_b^2(t)
```

关键代码：

```python
def otsu_threshold(gray):
    threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, threshold
```

特点：

- 优点：无需人工指定阈值。
- 缺点：仍然是全局阈值，对局部阴影不够鲁棒。

### 8.3 Sauvola 局部阈值

Sauvola 使用局部窗口内的均值和标准差计算每个位置的阈值：

```text
T(x,y) = m(x,y) × [1 + k × (s(x,y) / R - 1)]
```

其中：

- \(m(x,y)\)：局部窗口均值。
- \(s(x,y)\)：局部窗口标准差。
- \(k\)：调节局部对比度影响。
- \(R\)：标准差动态范围，本项目取 128。

局部均值和方差计算：

```text
mean = E[G]
variance = E[G^2] - E[G]^2
std = sqrt(variance)
```

关键代码：

```python
gray_float = gray.astype(np.float32)
mean = cv2.boxFilter(gray_float, -1, (window_size, window_size), normalize=True)
mean_sq = cv2.boxFilter(gray_float * gray_float, -1, (window_size, window_size), normalize=True)
variance = np.maximum(mean_sq - mean * mean, 0)
std = np.sqrt(variance)
threshold = mean * (1 + k * (std / R - 1))
binary = np.where(gray_float > threshold, 255, 0).astype(np.uint8)
```

参数选取：

- `SAUVOLA_WINDOW_SIZE = 25`
- `SAUVOLA_K = 0.2`
- `SAUVOLA_R = 128`

调试结果：

在主实验 11 张图像上，三种二值化方法平均指标如下：

| 方法 | 平均前景比例 | 平均边缘密度 |
|---|---:|---:|
| 固定阈值 | 0.2356 | 0.0587 |
| Otsu | 0.2467 | 0.0623 |
| Sauvola | 0.2725 | 0.0690 |

分析：

- Sauvola 的平均前景比例和边缘密度最高，说明它能保留更多局部文字细节。
- 固定阈值最简单，但遇到阴影时容易丢失浅文字。
- Otsu 在光照均匀时较好，但对局部光照变化适应性不如 Sauvola。

Sauvola 参数消融中，当 `window_size=25` 时：

| k | 平均前景比例 | 平均边缘密度 |
|---:|---:|---:|
| 0.2 | 0.2806 | 0.0884 |
| 0.3 | 0.2606 | 0.0847 |
| 0.5 | 0.2312 | 0.0797 |

说明 `k` 增大后前景文字比例下降，噪声可能减少，但弱文字也更容易被丢失。因此主流程采用 `k=0.2`，优先保留文字细节。

## 9. 二值图后处理模块

二值化结果可能包含孤立噪点或断裂笔画，因此使用形态学开运算和闭运算进行后处理。

开运算：

```text
open(Binary, B) = dilate(erode(Binary, B), B)
```

作用：去除小噪点。

闭运算：

```text
close(Binary, B) = erode(dilate(Binary, B), B)
```

作用：连接断裂笔画，填补小空洞。

关键代码：

```python
def postprocess_binary(binary, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed
```

参数选取：

- `kernel_size = 3`
- 选择原因：二值后处理只需要处理小噪点和小断裂，过大的核会破坏文字结构。

## 10. 参数调试总结

本项目的主要参数与调试依据如下：

| 参数 | 默认值 | 作用 | 调试依据 |
|---|---:|---|---|
| `MAX_WIDTH` | 1200 | 限制处理图像宽度 | 保证速度，同时保留足够细节 |
| `GAUSSIAN_KERNEL` | 5 | 高斯去噪 | 观察边缘图噪声和边界清晰度 |
| `CANNY_LOW` | 50 | Canny 低阈值 | 消融实验中折中边缘数量和噪声 |
| `CANNY_HIGH` | 150 | Canny 高阈值 | 与低阈值配合保留主要边界 |
| `SHADOW_KERNEL_SIZE` | 31 | 阴影背景估计 | 适合较大范围光照变化 |
| `MORPH_KERNEL_SIZE` | 15 | 顶帽/底帽增强 | 对比度和清晰度之间折中 |
| `FIXED_THRESHOLD` | 127 | 固定阈值二值化 | 灰度中间值，作为基准方法 |
| `SAUVOLA_WINDOW_SIZE` | 25 | 局部统计窗口 | 兼顾局部适应性和平滑性 |
| `SAUVOLA_K` | 0.2 | Sauvola 阈值调节 | 保留更多弱文字细节 |
| `SAUVOLA_R` | 128 | 标准差动态范围 | 8 位灰度图常用取值 |

调试过程主要通过以下输出观察：

- `outputs/03_edges/`：判断 Canny 边缘是否过多或过少。
- `outputs/04_contours/`：判断文档角点定位是否准确。
- `outputs/05_warped/`：判断透视矫正是否方向正常。
- `outputs/06_shadow_removed/`：判断阴影是否被削弱。
- `outputs/07_enhanced/`：判断文字对比度是否增强。
- `outputs/08_binary_fixed/`、`09_binary_otsu/`、`10_binary_sauvola/`：比较三种二值化结果。
- `outputs/metrics.csv`：用平均前景比例、边缘密度、对比度和清晰度辅助分析。

## 11. 本节可用于报告的总结

本项目的算法实现遵循“几何校正优先、光照增强其次、二值分割最后”的思路。Canny 边缘检测和轮廓多边形近似用于定位文档四角，四点透视变换用于完成倾斜校正；形态学闭运算用于估计背景并去除阴影，顶帽和底帽变换用于增强文字结构；固定阈值、Otsu 和 Sauvola 三种方法用于比较不同二值化策略。实验表明，默认参数能够完成 11 张文档图像的全流程处理，其中 Sauvola 在平均前景比例和边缘密度上高于固定阈值与 Otsu，更适合存在阴影和光照不均的文档场景。
