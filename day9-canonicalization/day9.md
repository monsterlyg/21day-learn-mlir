# Day 9：让 IR 自己变简单

## 今天我们要解决什么问题？

前面的 Pattern 练习已经让你亲手改写过 IR：找到一段计算，再把它替换成更合适的形式。Day 8 又让 `mynpu` 的 Op 能通过 Interface 告诉通用 Pass 自己的能力。

今天接着往前走一步：**一段改写规则写好以后，谁来调用它？什么时候调用？如果一次改写又制造出新的可简化内容怎么办？**

我们会从一个很小的例子开始：两层连续的 ReLU。

```text
relu(relu(x))
```

连续做两次 ReLU，结果和做一次相同。因此可以把 IR 简化成：

```text
relu(x)
```

这条规则很短，但它会带我们认识三个常见词：**fold、canonicalization、greedy rewrite**。学完之后，你应该能亲手给 `mynpu.relu` 加规则，并看着 IR 从两层变成一层。

## 先找回之前学过的 Pattern

先打开 [FusionPasses.cpp](/data8/liyinggang/Projects/CXX/mlir-learn/day6-pattern/FusionPasses.cpp)，找到 `FoldAddZeroPattern` 和 `FoldAddZeroPass`。

你会看到这条链：

```text
写 Pattern
   ↓
把 Pattern 放进 RewritePatternSet
   ↓
调用 applyPatternsGreedily
```

这和之前的融合规则是同一套基础设施。今天不是从头学一种完全不同的改写方式，而是继续回答两个问题：

1. 很简单的改写，能不能让 Op 自己提供？
2. 如何让 MLIR 的通用 `canonicalize` Pass 找到并调用这些规则？

## 第一步：准备一个“看起来有点多余”的 IR

在 `day8-interface` 的测试目录里准备这个输入：

```mlir
module {
  func.func @nested_relu(%x: tensor<4xf32>) -> tensor<4xf32> {
    %a = mynpu.relu %x, 0 : tensor<4xf32> -> tensor<4xf32>
    %b = mynpu.relu %a, 0 : tensor<4xf32> -> tensor<4xf32>
    return %b : tensor<4xf32>
  }
}
```

先别急着写 C++。观察一下：

- `%b` 是第二层 ReLU 的结果；
- 第二层 ReLU 的输入 `%a`，又是第一层 ReLU 的结果；
- 两层的 `axis` 和类型都相同。

问自己：如果把第二层 Op 删掉，直接返回 `%a`，函数结果会变吗？对这个例子来说不会。接下来要做的，就是把这个判断教给 MLIR。

## 第二步：把规则挂到 `mynpu.relu` 上

在 `day8-interface/IR/MyNPUOps.td` 里的 `MyNPU_ReluOp` 定义中加一行：

```tablegen
let hasCanonicalizer = 1;
```

这句话可以读成：“这个 Op 有自己的规范化规则。”TableGen 会生成 `ReluOp::getCanonicalizationPatterns(...)` 的声明。

然后在 `day8-interface/IR/MyNPUOps.cpp` 中实现规则：

```cpp
#include "mlir/IR/PatternMatch.h"

namespace {
struct RemoveNestedRelu : mlir::OpRewritePattern<mynpu::ReluOp> {
  using OpRewritePattern<mynpu::ReluOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mynpu::ReluOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // 看看当前 ReLU 的输入是不是另一个 ReLU 的结果。
    auto inner = op.getInput().getDefiningOp<mynpu::ReluOp>();
    if (!inner)
      return mlir::failure();

    // 配置不同，就先不改。
    if (op.getAxis() != inner.getAxis())
      return mlir::failure();

    // 替换结果的类型必须完全一致。
    if (op.getResult().getType() != inner.getResult().getType())
      return mlir::failure();

    // 外层 ReLU 的结果改用内层 ReLU 的结果。
    rewriter.replaceOp(op, inner.getResult());
    return mlir::success();
  }
};
} // namespace

void mynpu::ReluOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<RemoveNestedRelu>(context);
}
```

现在逐段读这段代码：

1. `OpRewritePattern<mynpu::ReluOp>` 表示这条规则只检查 `mynpu.relu`。
2. `getDefiningOp<ReluOp>()` 沿着输入 Value 找到“生产这个值的 Op”。找不到就说明不是连续 ReLU，返回 `failure()`。
3. 只有配置和结果类型都对得上，才执行改写。
4. `replaceOp` 把外层 ReLU 的结果使用者改接到内层结果，并删除外层 Op。

这里把条件检查放在改写之前。一个 Pattern 一旦返回 `failure()`，就不应该已经改动 IR。

> 访问器以当前 TableGen 生成结果为准。若 `getInput()` 或 `getAxis()` 的名称与你本地生成文件不同，查看 `day8-interface/build/MyNPUOps.h.inc` 中 `ReluOp` 的声明。

## 第三步：运行通用 `canonicalize`，看改写发生

重新构建 Day 8 插件后，用加载了 MyNPU Dialect 的 `mlir-opt` 运行 `canonicalize`。命令格式如下，工具路径按本机安装调整：

```bash
mlir-opt \
  --load-dialect-plugin=/path/to/day8-interface/build/libMyNPU.so \
  -p 'builtin.module(canonicalize)' \
  nested_relu.mlir
```

改写前有两个 `mynpu.relu`，改写后应该只剩一个，函数直接返回它的结果：

```mlir
module {
  func.func @nested_relu(%x: tensor<4xf32>) -> tensor<4xf32> {
    %a = mynpu.relu %x, 0 : tensor<4xf32> -> tensor<4xf32>
    return %a : tensor<4xf32>
  }
}
```

### 如果没有变化，按这个顺序检查

1. `.td` 中的 `hasCanonicalizer = 1` 是否加在 `MyNPU_ReluOp` 上？
2. C++ 实现的函数名和签名是否与生成的 `MyNPUOps.h.inc` 一致？
3. `mlir-opt` 是否加载了 MyNPU Dialect 插件？
4. 输入中第二个 ReLU 的确实是第一个 ReLU 的结果吗？`axis` 和类型是否匹配？
5. 当前执行的 pipeline 是否包含 `canonicalize`？

### 加一个负例

把第二层的 `axis` 改成不同值，重新观察结果。我们在规则里明确要求 axis 相同，因此这个输入应保持两层 ReLU。

负例的作用是确认规则有边界：它不是什么匹配到 ReLU 就随便删一个，而是在自己能证明安全时才改写。

## 回头理解：Canonicalization 是什么？

上面的规则叫 **canonicalization pattern**。用日常话说，它是给某个 Op 写的一条“整理方法”，让常见、重复或多余的写法变得更简单。

MLIR 有一个通用的 `canonicalize` Pass。它会收集已加载 Op 提供的规范化规则，反复尝试应用；因此 Pass 不需要写成“先找 ReluOp，再调用 RemoveNestedRelu”。Op 自己登记了规则，通用 Pass 就能发现它。

规范化不等于高级优化。它通常做局部、便宜、稳定的整理，例如：

- 常量计算：两个已知常量直接算出结果；
- 消除无效操作：移除恒等变换或重复计算；
- 合并相邻形式：把连续的兼容操作简化；
- 清理无用纯操作：结果没人使用、副作用又为空的 Op 可以删除。

MLIR 的 canonicalization 是 **best-effort（尽力而为）**：它努力应用规则，但不保证每段 IR 都有唯一的“标准答案”，也不保证所有可能的优化都能完成。[MLIR Canonicalization 文档](https://mlir.llvm.org/docs/Canonicalization/)

## 另一个词：Greedy Rewrite

为什么叫“贪心”？因为驱动器会拿一条当前能用的规则就先用，然后继续检查受影响的 Op。

可以把它想成整理桌面：

```text
看一个 Op
  ↓
找到可用的整理规则？── 否 ──→ 看下一个 Op
  │ 是
  ↓
执行改写
  ↓
重新检查受影响的 Op
  └─────────────────── 继续，直到没有可做的整理
```

所以 `applyPatternsGreedily` 不只是把输入 Op 扫描一遍。一次改写可能让周围的 IR 出现新的匹配机会，驱动器会再次检查相关部分。它会在没有更多改写时停止，也有迭代/改写上限，避免有缺陷的规则无限循环。[MLIR Pattern Rewriter 文档](https://mlir.llvm.org/docs/PatternRewriter/)

已有的 `FoldAddZeroPass` 是你见过的显式写法：Pass 创建自己的 `RewritePatternSet`，然后直接调用 `applyPatternsGreedily`。今天的 `-canonicalize` 则是通用整理 Pass：它从 Op 注册处收集规则，再使用类似的贪心驱动来应用它们。

## 那 `fold` 又是什么？

现在已经认识了 pattern 和驱动器，再看一个更小的问题：如果一个 Op 的结果只依赖常量，或者它其实等于某个已有输入，还要写完整的 `RewritePattern` 吗？

很多时候不需要。Op 可以实现 **fold**，把这种局部判断告诉 MLIR。

例如：

```text
arith.addi(2, 3) → 常量 5
arith.addi(x, 0) → x
```

可以把 fold 想成 Op 回答一个问题：“只看我自己、操作数的常量信息和已有 Value，我现在能不能直接给出结果？”

它和 Pattern 的主要区别是：

| | Fold | Rewrite Pattern |
|---|---|---|
| 观察范围 | 当前 Op 及其输入 | 可以沿 use-def 检查多个 Op |
| 能否新建 Op | 不能 | 可以通过 rewriter 创建 |
| 常见用途 | 常量计算、返回已有输入、局部恒等式 | 子图改写、融合、插入新计算 |
| 在哪里声明 | ODS 的 `hasFolder` | ODS canonicalizer hook 或某个 Pass 的 PatternSet |

单结果 Op 的 fold 常见返回类型是 `OpFoldResult`：可以返回一个已存在的 Value、一个表示常量的 Attribute，或空结果表示“现在不能 fold”。多结果 Op 的签名不同，查看 TableGen 生成的声明即可。

**选择时先问：我需要创建新 Op 吗？** 如果需要，写 Pattern；如果只需返回常量或已有 Value，优先考虑 fold。MLIR 文档也建议，能用 fold 表达的 canonicalization 优先放进 fold。[MLIR Canonicalization：fold 与 pattern](https://mlir.llvm.org/docs/Canonicalization/)

## 一个重要的语义提醒：浮点数不像普通实数

现有 `FoldAddZeroPattern` 把 `arith.addf(x, 0.0)` 改成 `x`。它很适合练习 `matchAndRewrite` 和 `replaceOp`，但**规则是否正确还要另外证明**。

浮点有 `+0.0` 和 `-0.0`。例如 `x` 是 `-0.0` 时，执行 `x + +0.0` 的结果可能与原来的 `x` 不同。若一个规则忽略了符号零，就可能把代码变短，却改变程序结果。

可以查看当前 LLVM checkout 的 `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` 中 `AddFOp::fold`：它处理右侧负零的恒等情形，并对常量输入进行计算。这个例子说明：canonicalization 的第一要求是语义相同；简洁只是结果，不是证明。

## 今天的练习路线

### 练习 A：看懂已有 Pattern

阅读 `day6-pattern/FusionPasses.cpp`：

1. 找到 `FoldAddZeroPattern` 的匹配条件。
2. 找到成功时调用的 rewriter API。
3. 找到 `FoldAddZeroPass` 如何创建 PatternSet 和启动贪心驱动。

做到这里，你应该能不看答案说出“谁写规则、谁选规则、谁反复应用规则”。

### 练习 B：让 `mynpu.relu` 自己提供规则

1. 在 `MyNPU_ReluOp` 加 `let hasCanonicalizer = 1;`。
2. 实现 `RemoveNestedRelu` pattern。
3. 用两层相同 axis 的 ReLU 运行 `canonicalize`。
4. 将 axis 改成不同值做负例。
5. 检查改写前后的结果类型一致。

### 练习 C：解释 fold 与 pattern

各用自己的话回答：

- 为什么“两常量相加”更适合 fold？
- 为什么“Conv + Add + ReLU 融合”需要 Pattern？
- 为什么 fold 不能随意创建新的计算 Op？

## 规范化规则的四条安全线

写规则时可以按这四个问题自检：

1. **语义一样吗？** 包括整数溢出、浮点 NaN/正负零、shape 和副作用。
2. **类型一样吗？** 替换结果的类型要满足原 Op 使用者的要求。
3. **会停下来吗？** 避免 `A → B → A` 这样的循环规则。
4. **改得够局部、够快吗？** Canonicalize 会反复运行；昂贵的全图搜索或硬件代价模型通常应放在独立优化 Pass。

还要记住：Pass pipeline 不应该依赖 canonicalize 才能保证正确性。Canonicalization 是清理机会，不是运行程序前必须执行的“修复步骤”。

## 今天结束时，试着讲给别人听

不看文档，按这个顺序解释一遍：

1. `relu(relu(x))` 为什么能变简单？
2. Op 怎么把规则登记给通用 canonicalize Pass？
3. Greedy driver 为什么需要反复检查？
4. fold 和 Pattern 的能力边界有什么不同？
5. 为什么 `x + 0.0 → x` 不能只用普通代数判断？

能讲清楚这五个问题，Day 9 的主线就已经掌握了。下一天转向 Symbol 和函数：那时关注点会从“一个 Op 怎样变简单”，转到“函数名如何被引用和解析”。

## 参考资料

- [MLIR Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
- [MLIR Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)
- [MLIR Toy Tutorial, Chapter 3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- 仓库代码：`day6-pattern/FusionPasses.cpp`、`day6-pattern/README.md`、`day8-interface/IR/MyNPUOps.td`、`day8-interface/IR/MyNPUOps.cpp`
