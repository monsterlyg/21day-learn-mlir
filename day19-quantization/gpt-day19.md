# Day 19：量化基础——从实数映射到整数

## 今天的目标

用数值例子理解 scale、zero point、rounding、clamp，以及 per-tensor/per-channel。再区分公式、校准和 IR 变换：公式定义映射，校准估计参数，IR 则需要表达参数并安排计算。

## 1. 仿射量化

常见写法：

```text
q = clamp(round(x / scale) + zero_point, qmin, qmax)
x_hat = scale * (q - zero_point)
```

`scale > 0`；zero point 是整数域中对应实数 0 的位置；clamp 把结果限制在目标整数范围。整数范围、舍入方式（特别是半整数）、对称/非对称约定必须按所用格式或工具链注明。

## 2. 手算例子

假设 signed INT8、`scale = 0.1`、`zero_point = 0`、范围 `[-128,127]`。对 `x = 0.26`：

```text
q = round(0.26 / 0.1) = 3
x_hat = 0.1 * (3 - 0) = 0.3
误差 = 0.3 - 0.26 = 0.04
```

再挑一个超出可表示范围的值，确认 clamp；挑一个半整数边界值，明确采用哪种舍入规则。

## 3. Per-tensor 与 per-channel

- **Per-tensor**：整个 tensor 共用 scale/zero point，参数简单；通道间范围差异大时精度可能受限。
- **Per-channel**：沿指定轴为每个通道设置 scale（有些约定也为每通道设置 zero point）。参数数量和轴必须与布局、通道顺序匹配。

查 IR 时不仅看参数数量，也要记下 quantization axis、layout 和权重通道顺序。

## 4. 校准和 IR 表示

PTQ 校准利用代表性输入估计范围和量化参数；公式给定参数后执行映射。QAT 则在训练中模拟量化影响。它们工作位置不同。

量化信息可能在专用类型、Op/属性、常量权重、quantize/dequantize 节点或项目 metadata 中。只看到 `i8` 不够，还要找到 scale、zero point、轴、舍入/饱和约定及计算域。

## 动手练习

### A. 数值表

用负值、零、正值、半整数边界和超范围值至少五项，计算 q、反量化值、误差。明确公式与舍入约定。

### B. 比较参数

对同一组值改变 scale/zero point，比较分辨率、范围和饱和次数。

### C. 查一个权重

从模型 IR 找权重类型和量化参数，核实是否 per-channel、轴对应哪维。没有真实 IR 时，使用明确标成假设的示例。

## 常见误会

- INT8 位宽本身不决定 scale。
- zero point 不一定是零。
- 校准是估计参数，量化公式使用参数。
- 多个 scale 仍可能配错轴。

## 自测

1. scale 和 zero point 各控制什么？
2. 校准与量化公式有什么区别？
3. 只看到 i8 为何无法判断量化语义完整？

## 参考资料

- [MLIR Quantization](https://mlir.llvm.org/docs/Quantization/)
- 具体约定请查所用 ONNX opset、TOSA 或编译器项目文档。
