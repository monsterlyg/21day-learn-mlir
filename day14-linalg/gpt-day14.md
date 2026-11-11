# Day 14：Linalg——用迭代空间表达张量计算

## 今天的目标

Day 12 区分了 tensor 值和 memref 存储。今天学习 Linalg 如何把计算的结构表达出来：迭代哪些维度、每个 operand 如何索引、哪些维度是并行迭代、哪些维度做归约。重点理解 `linalg.generic`，不追求记住所有 Op。

## 1. 读一个逐元素计算

下面是概念示意；具体语法以当前 MLIR 版本的测试和文档为准：

```mlir
#id = affine_map<(m, n) -> (m, n)>
linalg.generic {
  indexing_maps = [#id, #id],
  iterator_types = ["parallel", "parallel"]
} ins(%a : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
^bb0(%x: f32, %old: f32):
  %sum = arith.addf %x, %old : f32
  linalg.yield %sum : f32
} -> tensor<?x?xf32>
```

逐项读：两个维度都迭代；输入和输出都按相同坐标访问；区域把标量输入和输出初值相加，再 yield 结果。`outs` 的初值不是自动清零的保证。需要零初始化时，调用者必须提供有定义的零 tensor。

`parallel` 表示不同迭代点不通过该维度累积到同一个结果；`reduction` 则表示多个迭代点共同产生某个输出元素。

## 2. Indexing map 是坐标关系

Indexing map 把迭代坐标映射为 operand 坐标。常见关系：

```text
恒等访问：       (m, n) -> (m, n)
转置访问：       (m, n) -> (n, m)
行向量广播：     (m, n) -> (n)
矩阵乘法输出：   (m, n, k) -> (m, n)
```

map 说明“访问哪里”，不取代 verifier 的 shape 合法性检查。还要确认 operand 的维度、静态/动态信息和 map 相容。

## 3. 用迭代空间表示矩阵乘法

矩阵乘法 `C[m,n] += A[m,k] * B[k,n]` 的迭代变量是 `(m,n,k)`：

```text
A: (m, n, k) -> (m, k)
B: (m, n, k) -> (k, n)
C: (m, n, k) -> (m, n)
iterators: parallel, parallel, reduction
```

`k` 不出现在 C 的 map 中，因为不同 k 的结果累积到同一个 C[m,n]。累加要有语义正确的初始 C 值。对照阅读专用 `linalg.matmul`，理解高层 Op 如何比 generic 更直接地表达意图。

## 4. 阅读 Linalg Op 的检查顺序

1. 有几个迭代维度？各自是 parallel 还是 reduction？
2. 每个 operand 的 map 如何使用这些维度？
3. 输出是否遗漏归约维？
4. 区域参数与 operand 顺序如何对应？
5. yield 的类型是否与结果元素类型匹配？
6. 动态 shape 是否仍满足 operand 间的约束？

parallel 描述语义性质，不保证 lowering 一定生成并行代码。要判断可并行性，还应检查写入依赖、别名和 Op 的约束。

## 动手练习

### A. 找本机可用工具

运行 `mlir-opt --help | rg -i 'linalg|loops'`，记录版本及可用转换 pass。选一个随当前 checkout 提供的合法 Linalg 示例，保存输入、命令和输出；不要照抄其他版本的 pass 名称。

### B. 拆解 elementwise generic

列出迭代变量、maps、区域参数和输出初值。试着将一个 identity map 改成广播 map，检查 verifier 结果并解释 shape 条件。

### C. 手绘 matmul 坐标

先不写语法，画出 `(m,n,k)` 到 A/B/C 的映射，标出归约维，再找一段实际 IR 逐项验证。

## 常见误会

- `outs` 不保证零初始化；它可能也是区域计算的输入。
- `tensor.empty` 不会填零。
- Indexing map 描述坐标，不会自行确保所有 shape 都合法。
- “parallel”不承诺某个后端一定并行执行。

## 自测

1. iterator 和 indexing map 各回答什么问题？
2. Matmul 的 K 维为何是 reduction？
3. Linalg 的结构信息怎样帮助后续 lowering？

## 参考资料

- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Transform Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
