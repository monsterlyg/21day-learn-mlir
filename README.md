# MLIR 学习路线：从 IR 基础到 AI 编译器纵向切片

## 学习目标

这条路线面向已经完成 Day 1–8 初步练习、希望继续理解 MLIR 和 AI 编译器的人。目标不是 21 天内掌握 MLIR 全部功能，而是逐步具备三种能力：

1. 能读懂常见 MLIR IR、Dialect、Op、Pass 和 Rewrite。
2. 能为小型自定义 Dialect 实现并验证一段转换或推导逻辑。
3. 能沿一个神经网络算子，从模型导入一路追踪到目标相关 IR，并解释主要信息如何变化。

按每天约 2–3 小时安排；如果某个概念或练习没有跑通，就把计划拆成更多天，不必追赶日历。

## 先确认 Day 1–8 的学习边界

现有练习已经接触 MLIRContext、Operation/Value、Dialect、Pass、Pattern、ODS、自定义 Dialect、Dialect Conversion、OpInterface 和 InferTypeOpInterface。这里的“接触”不等于已经掌握：

- Day 4 的 walk 练习是手工遍历，不等于 MLIR AnalysisManager。
- Day 7 的主要转换是 `mynpu.addf → arith.addf`，还没有覆盖 TypeConverter 或复杂合法化。
- Day 8 的文档范围大于代码范围。以当前代码为准：Conv 有返回类型推导，Conv/Addf 有计算成本接口；未实现的 MemoryEffect/DialectInterface 内容属于阅读拓展。
- Day 6 的 `arith.addf(x, 0.0) → x` 是 pattern API 示例。浮点语义可能包含有符号零等边界，不能仅凭实数代数就认为这种改写总是正确；Day 9 会用它讨论语义审查。

## 每日安排

### 阶段 A：基础和已有项目（Day 1–8）

| 天 | 主题 | 可检查的产出 |
|---|---|---|
| 1 | Context、解析、Module/Operation、Location、walk | 修改输入 IR，并解释遍历访问到的 Op |
| 2 | SSA Value、OpResult、BlockArgument、Region/Block、Def-Use | 为一个小函数画出值的定义和使用关系 |
| 3 | Dialect、类型化 Op、`isa`/`dyn_cast`、基本类型 | 分类 `arith.addi`/`arith.mulf`；阅读 tensor、memref、linalg 示例 |
| 4 | Pass 生命周期、锚点、walk、插件 | 运行统计 Pass，并说明手工 walk 与 Analysis 的区别 |
| 5 | RewritePattern、PatternRewriter、贪心驱动、子图匹配 | 为 fusion pattern 准备成功和不应匹配的输入 |
| 6 | ODS/TableGen、自定义 Dialect、Op/Attr、parse/print/verify | 给 `mynpu.conv` 明确语义和错误条件，尝试补 verifier |
| 7 | ConversionTarget、ConversionPattern、partial/full conversion | 跑通 `mynpu.addf → arith.addf`，观察不合法 Op 如何导致 full conversion 失败 |
| 8 | Trait、OpInterface、InferTypeOpInterface、Memory Effect | 能指出接口定义、Op 实现、Pass 消费接口各在哪里；验证 Conv 推导路径 |

### 阶段 B：优化和 IR 结构（Day 9–13）

| 天 | 主题 | 可检查的产出 |
|---|---|---|
| 9 | [Folding、Canonicalization、Greedy Rewrite](day9-canonicalization/day9.md) | 看懂 fold 与 rewrite pattern 的边界；为 `mynpu.relu` 写一个保持语义的规范化规则 |
| 10 | [Symbol、SymbolRefAttr、函数和调用](day10-symbols/day10.md) | 构造两个函数及调用，区分 SSA Value 和 `@symbol`，查找符号引用 |
| 11 | [Region、BlockArgument、SCF 控制流](day11-scf/day11.md) | 画出带结果的 `scf.if` 和循环的控制流、值传递 |
| 12 | [Tensor/MemRef、静态/动态 Shape](day12-tensor-shape/day12.md) | 处理动态维度；解释 tensor 值和 memref 存储的差别 |
| 13 | [AnalysisManager 入门、分析失效](day13-analysis/day13.md) | 写一个只读统计 Analysis，由 Pass 读取；解释何时需要保留或失效分析 |

Day 13 只学 AnalysisManager 的基本模型即可；缓存、依赖分析和复杂失效规则可以延后。

### 阶段 C：结构化张量计算和模型 IR（Day 14–18）

| 天 | 主题 | 可检查的产出 |
|---|---|---|
| 14 | [Linalg、`linalg.generic`、迭代器和 indexing maps](day14-linalg/gpt-day14.md) | 表达 elementwise 算子与简化 MatMul，解释 parallel/reduction 维度 |
| 15 | [Bufferization、tensor 到 memref、别名与拷贝](day15-bufferization/gpt-day15.md) | 先观察一个小流程的输入/输出 IR，再解释为何某处需要分配或复制 |
| 16 | [高层 AI Dialect：按目标选择 TOSA 或 TPU-MLIR Top](day16-ai-dialect/gpt-day16.md) | 读一个 Conv/Elementwise Op 的语义和一个对应 lowering；TOSA 为可选比较项 |
| 17 | [模型导入和 IR 检查](day17-model-import/gpt-day17.md) | 用现成前端导入小 ONNX 模型，观察生成的高层 IR；不以自写 importer 为目标 |
| 18 | [图优化和 Pass Pipeline](day18-pipeline/gpt-day18.md) | 逐步运行 canonicalize、CSE、DCE 或项目自有 Pass，保存各阶段 IR |

One-Shot Bufferize 通常放在 tensor 优化之后、memref 后续 lowering 之前学习和运行；不要在还没理解 Linalg/tensor 语义时从内存规划细节开始。

### 阶段 D：量化与端到端项目（Day 19–21）

| 天 | 主题 | 可检查的产出 |
|---|---|---|
| 19 | [量化基础：scale、zero point、per-tensor/per-channel、校准](day19-quantization/gpt-day19.md) | 用一个数值例子解释 FP32 到 INT8 的映射和误差；区分量化公式与 IR 改写 |
| 20 | [TPU-MLIR 源码地图](day20-tpu-mlir-map/gpt-day20.md) | 选择一个 Op，记录它的定义、导入、优化、量化和 lowering 入口 |
| 21 | [小模型 Conv 纵向切片](day21-conv-slice/gpt-day21.md) | 保存导入后、高层优化后、量化后和目标相关 IR，并说明每阶段改变了什么 |

如果目标明确是 TPU-MLIR，Day 16 优先看其实际使用的 Top/TPU 表达和 pipeline；TOSA 作为对照阅读即可。Day 17 先把现有 `model_transform.py` 当作导入黑盒，先理解产物，不要同时承担实现 ONNX importer 的额外项目。

## 每天的工作方式

一次变换练习至少保留：

1. 输入 IR 和精确运行命令。
2. 变换前后 IR，或失败时的诊断信息。
3. 一个会改写的正例和一个必须保持不变的负例。
4. 对语义边界的说明：类型、shape、use-def、副作用、浮点/整数规则。

遇到 API 不确定时，优先查当前 checkout 的头文件、ODS 定义和测试；MLIR API 与命令行选项会随版本变化。

## 路线图

```text
IR 与 SSA（Day 1–3）
       ↓
Pass、Pattern、自定义 Dialect（Day 4–6）
       ↓
Dialect Conversion 与 Interface（Day 7–8）
       ↓
Canonicalization、Symbol、Control Flow、Tensor/Shape、Analysis（Day 9–13）
       ↓
Linalg、Bufferization、AI Dialect、模型 IR（Day 14–18）
       ↓
量化与 TPU-MLIR 纵向切片（Day 19–21）
```

核心闭环是选一个小模型并追踪 Conv：当前 IR 表达什么语义？哪些 shape、layout 或量化信息已经确定？运行了哪些 Pass？下一阶段要求什么合法 IR？

## 延后拓展

- TypeConverter、复杂 partial conversion 和 ConversionTarget 动态合法化。
- 深入 AnalysisManager 缓存、依赖分析和失效传播。
- 完整 One-Shot Bufferize、自定义 BufferizableOpInterface、内存规划。
- TOSA 全量覆盖、ONNX importer 实现、生产级 PTQ/QAT。
- TPU-MLIR 的所有芯片后端、代码生成和运行时。

优先完成一条小而可验证的端到端路径；只有当目标项目需要时，再扩展到这些主题。
