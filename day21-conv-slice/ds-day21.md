# Day 21：Conv 纵向切片——保存五个阶段的 IR

## 今天要解决什么问题？

前面每一天都在学一个局部机制。今天把它们串成一条能交付的路径：

> 选一个小模型的 Conv，从模型信息出发，依次保存「导入后、高层优化后、量化后（可选）、目标相关」的 IR，并说明每个阶段改变了什么。

这条路径的价值不在于「跑通」本身，而在于你能回答：

- 当前 IR 表达什么语义？
- 哪些 shape、layout、dtype 已经确定？
- 运行了哪些 Pass？
- 下一阶段要求什么合法 IR？

本机没有 TPU-MLIR，所以今天用 TOSA → Linalg → Buffer → Loops 做一条**可跑的等价纵向切片**；最后给出 TPU-MLIR 的对照命令模板。

## 环境与交付物

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
mkdir -p conv-slice
cd conv-slice
```

最终建议保留：

```text
conv-slice/
├── stage0_onnx.txt        # 模型侧信息
├── stage1_tosa.mlir       # 高层 IR
├── stage2_linalg.mlir     # 结构化张量 IR
├── stage3_memref.mlir     # buffer 化后
├── stage4_loops.mlir      # 循环级 IR
└── notes.md               # 每阶段观察记录
```

## 阶段 0：先记录模型信息

用 Day 17 的脚本导出一个 `Conv + ReLU`，把关键信息写进 `stage0_onnx.txt`：

```text
input:  NCHW 1x3x16x16
weight: OIHW 8x3x3x3
bias:   8
pads:   [1, 1, 1, 1]
strides:[1, 1]
dilations:[1, 1]
group:  1
output: NCHW 1x8x16x16
```

如果少了这一步，后面看到 `1x16x16x8` 时很难判断它对不对。

## 阶段 1：高层 IR

把 model 信息翻译成 TOSA（Day 17 已解释 layout）：

```mlir
module {
  func.func @conv_relu(%input: tensor<1x16x16x3xf32>,
                       %weight: tensor<8x3x3x3xf32>,
                       %bias: tensor<8xf32>) -> tensor<1x16x16x8xf32> {
    %zp = arith.constant dense<0.0> : tensor<1xf32>

    %conv = tosa.conv2d %input, %weight, %bias, %zp, %zp {
      acc_type = f32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>,
      stride = array<i64: 1, 1>
    } : (tensor<1x16x16x3xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x16x16x8xf32>

    %relu = tosa.clamp %conv {min_val = 0.0 : f32, max_val = 3.4e38 : f32}
      : (tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>

    return %relu : tensor<1x16x16x8xf32>
  }
}
```

保存并验证：

```bash
$MLIR_OPT --tosa-validate stage1_tosa.mlir -o /dev/null
```

此时 IR 的特点：

- op 数量少，语义清楚。
- layout 是 NHWC/OHWI。
- 还没有循环、内存分配和 buffer 别名。

## 阶段 2：结构化张量 IR

先做 named lowering，再把剩下的 elementwise 降成 generic：

```bash
$MLIR_OPT -p 'builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg))' \
  stage1_tosa.mlir -o stage2_linalg.mlir
```

为什么要两个 pass？因为它们的分工不同：

- `tosa-to-linalg-named` 只把能对应 named op 的算子降掉，例如 `tosa.conv2d → linalg.conv_2d_nhwc_fhwc`。
- `tosa.clamp` 没有 named 对应物，需要 `tosa-to-linalg` 降成 `linalg.generic`。

只跑第一个 pass 会留下 `tosa.clamp`，后面 bufferize 会报：

```text
error: op was not bufferized
```

本机实测，阶段 2 的结果包含：

```text
1 × tensor.pad            （把 input 从 1x16x16x3 变成 1x18x18x3）
1 × linalg.conv_2d_nhwc_fhwc
2 × linalg.generic        （bias 广播、clamp）
0 × tosa.*
```

观察记录：

- `tosa.conv2d` 的 `pad` 属性变成了显式的 `tensor.pad low[0,1,1,0] high[0,1,1,0]`。
- Conv 的 weight layout 变成 `FHWC`。
- bias 先广播成 `outs` 初值。
- `clamp` 变成逐元素 generic。
- 仍然是 tensor 语义，没有分配。

可以用统一命令统计 op：

```bash
grep -oE 'linalg\.[a-z0-9_]+' stage2_linalg.mlir | sort | uniq -c
```

## 阶段 3：Buffer 化

```bash
$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1})' \
  stage2_linalg.mlir -o stage3_memref.mlir
```

本机实测：出现 3 个 `memref.alloc`。它们分别对应：

- padding 后的输入 `1x18x18x3`；
- conv 的输出 `1x16x16x8`（bias 广播直接写进这块 buffer，没有单独分配）；
- clamp 的输出 `1x16x16x8`。

观察记录：

- 函数参数和返回值变成 memref。
- `tensor.empty` 消失，变成 `memref.alloc`。
- 出现 `memref.cast`（布局适配），不一定是 copy。
- 此时还没有 `memref.dealloc`；如果需要，补 `buffer-deallocation-pipeline`（Day 15）。

统计：

```bash
grep -c 'memref.alloc' stage3_memref.mlir
```

## 阶段 4：循环级 IR

```bash
$MLIR_OPT -p 'builtin.module(convert-linalg-to-loops)' \
  stage3_memref.mlir -o stage4_loops.mlir
```

本机实测规模：

```text
19 × scf.for
 9 × memref.load / memref.store
```

这里不再有 `linalg.*`，结构化信息已经变成显式循环。观察记录：

- 每个 linalg op 变成一个或多个嵌套循环。
- 索引计算通过 `arith` 完成。
- 内存访问变成 `memref.load/store`。
- 从这里开始，可以继续往 LLVM/目标后端走。

## 把整条链写成脚本

```bash
#!/usr/bin/env bash
set -euo pipefail

MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt

$MLIR_OPT --tosa-validate stage1_tosa.mlir -o /dev/null

$MLIR_OPT -p 'builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg))' \
  stage1_tosa.mlir -o stage2_linalg.mlir

$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1})' \
  stage2_linalg.mlir -o stage3_memref.mlir

$MLIR_OPT -p 'builtin.module(convert-linalg-to-loops)' \
  stage3_memref.mlir -o stage4_loops.mlir

echo 'stage2:'
grep -oE 'linalg\.[a-z0-9_]+' stage2_linalg.mlir | sort | uniq -c
echo 'stage3 allocs:'
grep -c 'memref.alloc' stage3_memref.mlir
echo 'stage4 loops:'
grep -c 'scf.for' stage4_loops.mlir
```

每一步都可以单独重跑。这比一条超长 pipeline 更容易定位问题。

## 对照：TPU-MLIR 的同类路径

如果你有 TPU-MLIR 环境，可以把阶段 1 换成它的导入产物：

```bash
# 1. 导入：ONNX → top 方言
python tools/model_transform.py \
  --model_name conv \
  --model_def conv.onnx \
  --input_shapes [[1,3,16,16]] \
  --mlir stage1_top.mlir

# 2. 跑高层优化 / 量化 / 后端 lowering
python tools/run_mlir.py --mlir stage1_top.mlir --chip bm1684x

# 具体参数以 tools/model_transform.py --help 和 run_mlir.py --help 为准
```

对应的阶段名称会变成：

```text
top 方言 IR → 优化后 top IR → tpu 方言 IR → 目标代码
```

不要照抄命令里的模型名、chip 或输出目录，先 `--help` 确认。

## 交付记录模板

每个阶段至少记录这些字段：

| 阶段 | 文件 | 方言 | 关键 op | shape/layout | 内存 | 运行的 Pass |
|---|---|---|---|---|---|---|
| 0 | stage0_onnx.txt | ONNX | Conv, Relu | NCHW 1x3x16x16 | 无 | 导出 |
| 1 | stage1_tosa.mlir | tosa | conv2d, clamp | NHWC 1x16x16x3 | 无 | validate |
| 2 | stage2_linalg.mlir | linalg | tensor.pad, conv_2d_nhwc_fhwc, generic | NHWC | 无 | tosa-to-linalg-named, tosa-to-linalg |
| 3 | stage3_memref.mlir | memref/linalg | alloc, linalg, linalg.map | NHWC | 3 alloc | one-shot-bufferize |
| 4 | stage4_loops.mlir | scf/memref | scf.for, load/store | 显式索引 | 3 alloc | convert-linalg-to-loops |

## 今天的练习路线

### 练习 A：改 stride

把 stride 从 `[1,1]` 改成 `[2,2]`，pad 保持 `[1,1,1,1]`：

1. 手算输出 shape。
2. 从阶段 1 重新跑到阶段 4。
3. 比较阶段 4 的循环边界和 `memref.alloc` 的 shape。

### 练习 B：加一个算子

在 clamp 后面加一个 `tosa.add`（和另一个同 shape 输入逐元素相加），重新跑完整条链。记录阶段 2 的 generic 数量变化。

### 练习 C：只跑 named lowering

故意只跑 `tosa-to-linalg-named`，然后直接 bufferize，记录错误信息。解释为什么必须补 `tosa-to-linalg`。

### 练习 D：负例

把 weight 的 layout 从 `8x3x3x3` 改成 `3x3x3x8`（模拟 OIHW 没转置），运行 `--tosa-validate`，记录错误。用「layout 契约」解释为什么这里会失败。

## 常见坑

- **只跑 named lowering**：剩余 elementwise op 会缺 lowering，bufferize 报 `op was not bufferized`。
- **layout 混用**：NCHW 和 NHWC 的 shape 数字不能直接照抄。
- **把 alloc 数量当成性能结论**：它只是当前 pipeline 的结果，后续 deallocation/hoisting 会改变布局。
- **不看阶段规模**：`scf.for` 和 `memref.load` 数量是后续优化的基线。
- **TPU-MLIR 命令照抄**：chip、模型名、calibration 参数都要按本地环境确认。

## 今天结束时，试着讲给别人听

1. 五个阶段分别解决了什么问题？
2. 为什么 named lowering 之后还要 `tosa-to-linalg`？
3. bufferize 后新增的 `memref.alloc` 对应哪些 tensor？
4. `convert-linalg-to-loops` 之后丢失了哪些结构化信息？
5. 一条纵向切片最少要记录哪些字段，才能让别人复现？

## 参考资料

- [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [TOSA Specification](https://www.mlplatform.org/tosa/tosa_spec.html)
- [TPU-MLIR](https://github.com/sophgo/tpu-mlir)
- 本机源码：`mlir/lib/Conversion/TosaToLinalg/`、`mlir/lib/Dialect/Bufferization/Transforms/`
