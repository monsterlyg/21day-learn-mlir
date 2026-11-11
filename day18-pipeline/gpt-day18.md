# Day 18：图优化与 Pass Pipeline——逐阶段读 IR

## 今天的目标

逐个运行常见 pass，保存输入与输出 IR，建立变化与 pass 之间的因果关系。Canonicalize、CSE、DCE 的目标不同；它们也不会自动保证全局最优。

## 1. 三类常见优化

- **Canonicalize**：应用 Op 注册的规范化和 folding 规则，尝试形成更规范的 IR。
- **CSE**：合并满足等价性条件的公共子表达式；副作用、属性、类型和支配关系都重要。
- **DCE**：移除不可观察、无副作用且结果无人使用的计算。无结果 Op 仍可能有副作用。

先用 `mlir-opt --help | rg -i 'canonicalize|cse|dce'` 确认工具支持的 pass 名称。不同版本可能没有独立通用 DCE pass，切勿假定参数存在。

## 2. 每次只观察一个 pass

按本机帮助运行示意命令：

```sh
mlir-opt imported.mlir --canonicalize -o after-canonicalize.mlir
mlir-opt imported.mlir --cse -o after-cse.mlir
```

每次记录精确输入、命令和输出，并标出新增/删除/合并/改写的 Op。区分语义变化与纯打印形式变化。随后再组合成 `canonicalize → cse → 项目 pass`，保留每阶段快照。

Pass 顺序可能影响匹配机会：前一个 pass 形成的形式可能使后一个 pass 更容易工作。项目 pipeline 还可能根据配置选择不同分支。读源码时追踪 pass 实现、注册和实际 pipeline 调用。

## 3. 检查每个改写是否合法

检查类型、shape、副作用、别名、结果 use，以及浮点/整数语义。特别是浮点优化：实数代数恒等式不必然是浮点 IR 的无条件恒等式；需检查舍入、NaN、有符号零、快速数学等语义条件。

## 动手练习

### A. 构造小例子

准备重复计算、可折叠常量和无用纯计算，先预测各 pass 的效果，再比较输出。

### B. 保存模型快照

对 Day 17 的模型逐个 pass 运行，保存至少两个阶段，并说明变化由哪个 pass 导致。

### C. 对比顺序

在安全的小例子中比较 canonicalize→CSE 与 CSE→canonicalize。若结果相同，也说明不能据此认为所有 pipeline 顺序等价。

## 常见误会

- Canonicalize 不是全局最优求解器。
- CSE 不能仅凭文本相同合并所有 Op，副作用与支配关系会影响合法性。
- 无结果不等于无副作用，因此 DCE 不能随意删 Op。

## 自测

1. 三类优化分别解决什么问题？
2. 为什么先单独运行每个 pass？
3. 判断浮点改写时要检查哪些边界？

## 参考资料

- [MLIR Passes](https://mlir.llvm.org/docs/Passes/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/)
