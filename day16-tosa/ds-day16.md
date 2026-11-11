# Day 16：高层 AI Dialect——从 TOSA 到 Linalg

## 今天要解决什么问题？

Day 14 你学会了用 Linalg 表达计算，Day 15 学会了把它变成内存操作。但真实模型导入后，第一站通常不是 `linalg.generic`，而是某个**高层算子方言**：TOSA、TPU-MLIR 的 Top、ONNX 方言等。

为什么需要这一层？因为模型里的 `Conv`、`Pool`、`Resize` 有明确的算子语义和形状规则，先用高层 op 保住这些信息，再逐步 lowering，比直接摊成循环更容易验证和优化。

今天用本机自带的 **TOSA** 做样本，目标：

1. 读懂 `tosa.conv2d` 的操作数、属性和 layout。
2. 用 `tosa-validate` 和 `tosa-infer-shapes` 检查它。
3. 观察它如何 lowering 成 `linalg.conv_2d_nhwc_fhwc`。
4. 理解 named lowering 和 generic lowering 的区别。

## 环境

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
```

### 先确认版本差异

不同 TOSA 版本的 op 集合和属性会变化。本机这一版：

- `tosa.conv2d` 有 5 个操作数：input、weight、bias、input_zp、weight_zp。
- 没有 `tosa.relu`（直接写会报 `custom op 'tosa.relu' is unknown`）。
- ReLU 用 `tosa.clamp` 表达，最小值为 0。

这不是「TOSA 写错了」，而是版本差异。遇到 Op 不存在时，先查本机 `TosaOps.td`，不要照抄旧文档。

## 第一站：`tosa.conv2d` 的形状契约

一个最小例子：

```mlir
module {
  func.func @conv(%arg0: tensor<1x4x4x4xf32>,
                  %arg1: tensor<8x1x1x4xf32>,
                  %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    %zp = arith.constant dense<0.0> : tensor<1xf32>
    %0 = tosa.conv2d %arg0, %arg1, %arg2, %zp, %zp {
      acc_type = f32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>,
      stride = array<i64: 1, 1>
    } : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
    return %0 : tensor<1x4x4x8xf32>
  }
}
```

逐项读：

- **input**：`[N, H, W, C]`，这里是 `1x4x4x4`。
- **weight**：TOSA 使用 `[OC, KH, KW, IC]`，这里 `8x1x1x4`。
- **bias**：长度等于输出通道 `8`。
- **input_zp / weight_zp**：零点是**操作数**，不是属性。浮点模型通常传 0。
- **`pad`**：四个值，顺序是 `[top, bottom, left, right]`。
- **`stride` / `dilation`**：各两个值，`[H, W]`。
- **`acc_type`**：累加类型。

输出 shape 的计算和 Day 8 的 Conv 一样：

```text
OH = (IH + pad_top + pad_bottom - dilation_H * (KH - 1) - 1) / stride_H + 1
OW = (IW + pad_left + pad_right - dilation_W * (KW - 1) - 1) / stride_W + 1
```

本例 `pad=0, stride=1, KH=KW=1`，所以输出仍是 `4x4`，通道变成 8。

## 第二站：先验证，再转换

运行（把上面的 module 保存为 `conv.mlir`）：

```bash
$MLIR_OPT --tosa-validate conv.mlir
```

输入合法时，命令正常输出 IR。把输出类型故意写成 `tensor<1x4x4x7xf32>`，会得到：

```text
error: 'tosa.conv2d' op bias channels expected to be equal to output channels (7) or 1, got 8
```

这说明 TOSA 的 verifier 在检查通道契约，而不是只做语法解析。转换前先 validate，可以避免把错误带进 lowering。

## 第三站：shape inference 会做什么？

把输出类型写成动态：

```mlir
func.func @conv(...) -> tensor<?x?x?x?xf32>
```

运行：

```bash
$MLIR_OPT --tosa-infer-shapes <<'EOF'
...输出类型为 tensor<?x?x?x?xf32> 的 module...
EOF
```

会看到：

```mlir
%0 = tosa.conv2d ... -> tensor<1x4x4x8xf32>
%cast = tensor.cast %0 : tensor<1x4x4x8xf32> to tensor<?x?x?x?xf32>
return %cast : tensor<?x?x?x?xf32>
```

要点：

- 推理只在信息足够时把 `?` 变成具体数字。
- 为了同时满足「更精确的 op 结果」和「原来的函数签名」，它插入 `tensor.cast`。
- 这和 Day 12 的原则一致：能推的推，推不出的保留动态。

## 第四站：lowering 到 named Linalg

TOSA 的 lowering pass 是 **FunctionOpInterface** pass，不能直接挂在 `builtin.module` 上。正确写法：

```bash
$MLIR_OPT -p 'builtin.module(func.func(tosa-to-linalg-named))' conv.mlir
```

输出核心部分：

```mlir
%0 = tensor.empty() : tensor<1x4x4x8xf32>
%1 = linalg.generic {indexing_maps = [#map, #map1],
                     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg2 : tensor<8xf32>) outs(%0 : tensor<1x4x4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
} -> tensor<1x4x4x8xf32>

%2 = linalg.conv_2d_nhwc_fhwc {
  dilations = dense<1> : tensor<2xi64>,
  strides = dense<1> : tensor<2xi64>
} ins(%arg0, %arg1 : tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>)
  outs(%1 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
```

观察：

1. bias 先被广播成一个 generic，作为 conv 的 `outs` 初值。
2. conv 变成 named op `linalg.conv_2d_nhwc_fhwc`，weight layout 是 `[F, H, W, C]`。
3. TOSA 的 `pad` 没有出现在 named op 属性里，因为这个例子 pad 为 0。带 padding 时 lowering 会先插入显式的 `tensor.pad`，例如 `pad = [1,1,1,1]` 会生成：

```mlir
%padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] { ... }
    : tensor<1x16x16x3xf32> to tensor<1x18x18x3xf32>
```

然后 conv 在 padding 后的张量上执行。

Named lowering 的优点是保留算子语义，后续可以匹配 `linalg.conv_2d_*` 做 tiling 或转目标库。

## 第五站：generic lowering

另一条路：

```bash
$MLIR_OPT -p 'builtin.module(func.func(tosa-to-linalg))' input.mlir
```

`tosa-to-linalg` 会把更多 TOSA op 展开成 `linalg.generic` 和基础张量操作，而不是 named conv。它适合：

- 目标后端不认识 named op，只认识 generic；
- 想用统一的 elementwise 融合/分析处理所有算子。

取舍可以记成一句话：

```text
named 保留意图，generic 统一形态。
先 named 再按需 generalize，通常比一开始就全 generic 更好读。
```

## 第六站：TOSA 里的 ReLU

本机版本没有 `tosa.relu`。等价的表达是：

```mlir
%0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 3.4e38 : f32}
    : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
```

`clamp` 的语义是 `min(max(x, min_val), max_val)`，把下界设成 0 就是 ReLU。

这是一个很好的「读源码而不是背文档」的例子：先 `grep` 本机 `TosaOps.td`，再决定怎么写。

## 今天的练习路线

### 练习 A：手算输出 shape

把 conv 的 `stride` 改成 `[2, 2]`、`pad` 改成 `[1, 1, 1, 1]`，输入仍是 `1x4x4x4`，weight 是 `8x3x3x4`：

1. 手算 OH/OW。
2. 写进 IR，用 `--tosa-validate` 验证。
3. 用 `--tosa-infer-shapes` 检查推断结果。

### 练习 B：读 lowering

对上面的 conv 运行 `tosa-to-linalg-named`，回答：

1. bias 被放到哪里？
2. `linalg.conv_2d_nhwc_fhwc` 的 weight 维序是什么？
3. `outs` 的初值来自哪里？

### 练习 C：换一种算子

把 conv 换成 `tosa.depthwise_conv2d` 或 `tosa.matmul`，先查 `TosaOps.td` 的参数，再写一个最小例子。不要凭记忆猜属性。

### 练习 D：负例

把 weight 的输入通道从 4 改成 3（与 input 不匹配），运行 `--tosa-validate`，记录错误信息。这个错误应该在转换之前就被抓住。

## 常见坑

- **pipeline anchor 写错**：`tosa-to-linalg-named` / `tosa-to-linalg` 是 FunctionOpInterface pass，必须写 `builtin.module(func.func(...))`。
- **layout 混用**：ONNX/PyTorch 常见 NCHW，TOSA 是 NHWC，weight 是 OHWI。
- **pad 顺序**：TOSA 是 `[top, bottom, left, right]`，不是 `[H, W]` 两个值。
- **零点是属性**：本版本是操作数，不能写成 attribute。
- **照着旧文档写 op**：TOSA 版本间有增删，先查本机 `TosaOps.td`。

## 今天结束时，试着讲给别人听

1. 为什么模型 IR 需要高层算子方言？
2. `tosa.conv2d` 的 input/weight layout 分别是什么？
3. `tosa-validate` 和 `tosa-infer-shapes` 各解决什么问题？
4. named lowering 和 generic lowering 的取舍是什么？
5. 为什么 pipeline anchor 会影响 pass 能不能运行？

## 参考资料

- [TOSA Specification](https://www.mlplatform.org/tosa/tosa_spec.html)
- [MLIR TOSA Dialect](https://mlir.llvm.org/docs/Dialects/TOSA/)
- 本机源码：`mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td`、`mlir/lib/Conversion/TosaToLinalg/`
- 本机测试：`mlir/test/Dialect/Tosa/`、`mlir/test/Conversion/TosaToLinalg/`
