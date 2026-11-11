# Day 19：量化——从浮点到整数，再回到浮点

## 今天要解决什么问题？

量化不是「把 `f32` 改成 `i8`」这么简单。它是一组带参数的数值映射：

> 一个浮点值 `x`，怎样用有限的整数 `q` 表示？反过来，拿到 `q` 怎样近似恢复 `x`？误差有多大？

AI 编译器里，量化会同时影响**类型、IR 结构、算子语义和校准数据**。今天先抓住最核心的数学和 IR 表达，暂不展开生产级 PTQ/QAT 流程。

今天的目标：

1. 掌握仿射量化的两个公式：量化与反量化。
2. 用 `arith` + `math` 手写一个 fake-quant，理解每一步在做什么。
3. 认识 `quant` dialect 的类型和 `qcast`/`dcast`/`scast`。
4. 用 `--lower-quant-ops` 看量化类型如何展开成普通算术。

## 环境

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
```

## 第一站：两个公式

仿射量化（asymmetric affine quantization）：

```text
量化：   q = clamp(round(x / s) + zp, qmin, qmax)
反量化： x_hat = s * (q - zp)
```

三个参数：

- **scale `s`**：一个整数单位代表多少浮点值。
- **zero point `zp`**：浮点 0 对应的整数编码。
- **qmin/qmax**：存储类型的范围，例如 int8 是 `[-128, 127]`。

`s` 通常由数值范围决定：

```text
s = (max_val - min_val) / (qmax - qmin)
zp = round(qmin - min_val / s)
```

`s` 越大，能表示的范围越大，但每个台阶越粗；`s` 越小，精度越高，但更容易饱和。

### 一个手算例子

取 `s = 0.0078125`（即 `1/128`）、`zp = 0`、int8：

```text
x = 0.1
x / s = 12.8
roundeven(12.8) = 13
q = 13
x_hat = 13 * 0.0078125 = 0.1015625
error = 0.0015625，约 1.56%
```

再取 `s = 0.5`、`zp = 2`：

```text
x = 1.1
x / s = 2.2
roundeven(2.2) = 2
q = 2 + 2 = 4
x_hat = 0.5 * (4 - 2) = 1.0
error = 0.1
```

注意第二步的 `+ zp` 和反量化时的 `- zp` 必须成对出现，否则零点会漂移。

## 第二站：用普通算术写 fake-quant

在 MLIR 里，可以把量化过程拆成 `arith` + `math`：

```mlir
module {
  func.func @fake_quant(%x: tensor<4xf32>) -> tensor<4xf32> {
    %inv_scale = arith.constant dense<128.0> : tensor<4xf32>
    %zp        = arith.constant dense<0.0>   : tensor<4xf32>
    %qmin      = arith.constant dense<-128.0> : tensor<4xf32>
    %qmax      = arith.constant dense<127.0>  : tensor<4xf32>
    %scale     = arith.constant dense<0.0078125> : tensor<4xf32>

    %scaled  = arith.mulf %x, %inv_scale : tensor<4xf32>
    %rounded = math.roundeven %scaled : tensor<4xf32>
    %shifted = arith.addf %rounded, %zp : tensor<4xf32>
    %lo      = arith.maximumf %shifted, %qmin : tensor<4xf32>
    %hi      = arith.minimumf %lo, %qmax : tensor<4xf32>
    %dq      = arith.mulf %hi, %scale : tensor<4xf32>
    return %dq : tensor<4xf32>
  }
}
```

对应关系：

```text
x / s        → arith.mulf（乘 1/s）
round        → math.roundeven
+ zp         → arith.addf
clamp        → arith.maximumf + arith.minimumf
* s          → arith.mulf
```

为什么用 `math.roundeven` 而不是随便截断？因为「四舍五入到最近、平局取偶」是量化里常用的确定性规则，能减少系统性偏置。截断（toward zero）会引入方向性误差。

这个 IR 是 **fake quant**：它仍然用 f32 计算，但数值上模拟了「量化再反量化」的误差。它常用于 QAT 训练和校准验证。

## 第三站：`quant` dialect 怎么表达

MLIR 的 `quant` dialect 把「存储类型 + scale + zero point」编码进类型。例如：

```mlir
!qalias = !quant.uniform<i8:f32, 0.5:2>
```

读法：

- `i8`：实际存储是 8 位整数。
- `f32`：它近似表示的浮点类型。
- `0.5`：scale。
- `2`：zero point。

per-channel（每个通道一个 scale）：

```mlir
!qalias = !quant.uniform<i8:f32:1, {0.01, 0.1, 0.2}>
```

中间的 `:1` 是**量化维**（quantized dimension），后面的数组是每个通道的 scale。用的时候要注意张量 rank 和量化维必须匹配，否则 verifier 会报错。

三个核心 op：

```mlir
%q = quant.qcast %x : tensor<4x3xf32> to tensor<4x3x!quant.uniform<i8:f32, 0.5:2>>
%f = quant.dcast %q : tensor<4x3x!quant.uniform<i8:f32, 0.5:2>> to tensor<4x3xf32>
%s = quant.scast %q : tensor<4x3x!quant.uniform<i8:f32, 0.5:2>> to tensor<4x3xi8>
```

- `qcast`：浮点 → 量化类型。
- `dcast`：量化类型 → 浮点。
- `scast`：量化类型 ↔ 底层存储整数，只改「看法」，不做数值换算。

注意：`qcast`/`dcast` 是**数值转换**，`scast` 是**存储类型转换**。读 IR 时不要混淆。

## 第四站：`--lower-quant-ops` 展开了什么？

输入：

```mlir
!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @dcast(%arg0: !qalias) -> f32 {
  %0 = quant.dcast %arg0 : !qalias to f32
  return %0 : f32
}
```

运行（把上面的 IR 保存为 `dcast.mlir`）：

```bash
$MLIR_OPT --lower-quant-ops dcast.mlir
```

输出核心：

```mlir
%0 = quant.scast %arg0 : !quant.uniform<i8:f32, 2.000000e+00:10> to i8
%cst = arith.constant 2.000000e+00 : f32
%c10_i8 = arith.constant 10 : i8
%1 = arith.sitofp %0 : i8 to f32
%2 = arith.sitofp %c10_i8 : i8 to f32
%3 = arith.subf %1, %2 : f32
%4 = arith.mulf %3, %cst : f32
```

这正是反量化公式 `x_hat = s * (q - zp)`：

```text
q  → scast 到 i8
q、zp → sitofp 到 f32
(q - zp) → arith.subf
* s → arith.mulf
```

per-channel 的例子会展开成 `linalg.generic`，因为每个通道的 scale/zero point 不同，需要按通道索引取值。

## 第五站：per-tensor 和 per-channel 的差别

假设有两个通道：

```text
channel 0: x = 0.01
channel 1: x = 10.0
```

### per-tensor

如果用一个全局 scale `s = 10 / 127 ≈ 0.0787`：

```text
channel 0: q = round(0.01 / 0.0787) = round(0.127) = 0
           x_hat = 0，误差 100%
channel 1: q = round(10 / 0.0787) = 127
           x_hat ≈ 10，误差很小
```

通道 0 的数值被「大范围」挤没了。

### per-channel

如果两个通道各自定 scale：

```text
channel 0: s0 = 0.0001  → q = 100 → x_hat = 0.01
channel 1: s1 = 0.1     → q = 100 → x_hat = 10.0
```

两边都能保留。这就是 per-channel 在卷积权重上常用的原因：不同输出通道的数值范围可能差很多。

代价是：

- scale/zero point 从标量变成数组；
- 反量化需要按通道索引；
- IR 里会出现 `linalg.generic` 或类似的广播结构。

## 今天的练习路线

### 练习 A：手算

对下面的输入，手算 `q` 和 `x_hat`，写出误差：

```text
s = 0.01, zp = 5, x = 0.234
s = 0.25, zp = -3, x = 2.6
```

然后用第二站的 fake-quant IR 验证你的结果（改常量和输入）。

### 练习 B：看展开

把 `!quant.uniform<i8:f32, 0.5:2>` 的标量 `dcast` 跑一遍 `--lower-quant-ops`，逐行对应到反量化公式。

### 练习 C：per-channel

写一个 `tensor<2x3x!quant.uniform<i8:f32:1, {0.01,0.1,0.2}>>` 的 `dcast`，确认：

1. 量化维是 1；
2. scale 数组长度等于该维大小；
3. 展开后是 `linalg.generic` 而不是纯标量运算。

### 练习 D：负例

故意写一个 scale 为 0、或 zero point 超出 int8 范围的量化类型，运行 `mlir-opt`，记录 verifier 的诊断。理解「类型合法」和「数值合理」是两件事。

## 常见坑

- **`round` 和 `roundeven` 混用**：平局规则不同，误差分布不同。
- **忘记 clamp**：超出 `qmin/qmax` 的值会回绕或产生未定义行为。
- **`scast` 当成反量化**：它只改存储解释，不做 `s * (q - zp)`。
- **per-channel 维写错**：量化维必须小于张量 rank，scale 数组长度必须匹配。
- **scale 为 0**：除法/映射失去意义。
- **只看类型不看布局**：NHWC 下 per-channel 的通道维通常是最后一维，NCHW 下不是。
- **把 fake quant 当成真正量化**：fake quant 仍然用浮点算，只是模拟误差。

## 今天结束时，试着讲给别人听

1. 仿射量化的两个公式分别是什么？
2. zero point 为什么在量化时加、反量化时减？
3. `qcast`、`dcast`、`scast` 的区别是什么？
4. 为什么卷积权重常用 per-channel 量化？
5. `--lower-quant-ops` 之后，量化信息还剩在哪里？

## 参考资料

- [MLIR Quantization Dialect](https://mlir.llvm.org/docs/Dialects/QuantDialect/)
- [MLIR Quantization](https://mlir.llvm.org/docs/Quantization/)
- 本机源码：`mlir/include/mlir/Dialect/Quant/IR/QuantTypes.h`、`mlir/lib/Dialect/Quant/Transforms/LowerQuantOps.cpp`
- 本机测试：`mlir/test/Dialect/Quant/`
