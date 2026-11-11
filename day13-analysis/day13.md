# Day 13：Analysis——先看清 IR，再决定怎么改

## 从前几天的问题出发

前几天我们先学会了读函数名和调用，再读 Region、循环和动态 Shape。现在假设你要写一个优化 Pass：

> 这个 Module 里有多少个 `mynpu.conv`？

你当然可以在 Pass 里 walk 一遍 IR，数出来。那如果三个 Pass 都需要同一个统计结果呢？每个 Pass 都重复遍历可以工作，但计算重复；更麻烦的是，如果统计信息被缓存，IR 改了以后，旧结果还能用吗？

MLIR 的 **Analysis** 就是用来回答这类问题的：它根据某个 Operation 的当前 IR 计算信息，自己不改 IR；Pass 按需请求它，PassManager 负责管理和失效缓存。

今天只学 AnalysisManager 的基本使用，不深入复杂依赖图和自定义 invalidation。

## 第一站：Pass 和 Analysis 的分工

先用一句话区分：

- **Pass**：读取 IR，也可以改写 IR；它负责做一件变换或检查工作。
- **Analysis**：读取 IR、计算并保存信息；它不改 IR。

例如：

```text
ConvCountAnalysis：这个 Module 里有几个 Conv？
MyOptimizationPass：根据 Conv 数量决定是否执行某种优化。
```

Analysis 不会自己启动 Pass，也不负责安排整个流水线。PassManager 调度 Pass；Pass 在需要时向当前 Operation 的 AnalysisManager 请求 Analysis。

## 第二站：先写一个最简单的 Analysis

以统计 `mynpu.conv` 数量为例：

```cpp
struct ConvCountAnalysis {
  explicit ConvCountAnalysis(mlir::Operation *op) {
    op->walk([&](mlir::Operation *nested) {
      if (mlir::isa<mynpu::ConvOp>(nested))
        ++count;
    });
  }

  size_t getCount() const { return count; }

private:
  size_t count = 0;
};
```

读这段代码：

1. 构造函数接收需要分析的 Operation；这里传入的通常是 Pass 当前处理的 Module。
2. 构造函数只 walk 并计数，不修改任何 IR。
3. 计数结果保存在对象里，调用者通过 `getCount()` 查询。

Analysis 的构造函数会在 Analysis 第一次被请求时运行。之后在结果仍有效的情况下，AnalysisManager 可以复用缓存的对象，不必每次都重新 walk。

## 第三站：Pass 怎么请求它？

在 OperationPass 中可以调用 `getAnalysis<AnalysisT>()`：

```cpp
struct ReportConvCountPass
    : mlir::PassWrapper<ReportConvCountPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportConvCountPass)

  void runOnOperation() override {
    const auto &analysis = getAnalysis<ConvCountAnalysis>();
    llvm::outs() << "Conv count = " << analysis.getCount() << "\n";

    // 此 Pass 只读 IR，因此它计算的 Analysis 仍然有效。
    markAllAnalysesPreserved();
  }
};
```

第一次运行到 `getAnalysis<ConvCountAnalysis>()` 时，如果当前 Module 还没有这个分析结果，AnalysisManager 会构造它；如果缓存里已有有效结果，就可以复用。

这和 Day 4 的手工 `module.walk(...)` 有什么不同？手工 walk 是一次性遍历；Analysis 把结果封装成可请求、可缓存、可失效的分析对象，便于多个 Pass 在结果有效时复用。

## 第四站：缓存为什么要失效？

假设初始 Module 里有 2 个 Conv：

```text
ConvCountAnalysis 计算结果：2
```

接下来某个 Pass 删除一个 Conv。如果缓存仍然返回 2，就错了；对新 IR 重新计算才应该得到 1。

因此 MLIR 默认采取保守规则：**Pass 运行后，Analysis 通常视为可能失效；只有 Pass 确认某些分析仍然正确时，才标记保留。**

最容易理解的情况是只读 Pass：它没有修改 IR，因此可以调用 `markAllAnalysesPreserved()`。如果 Pass 改了 IR，就不能因为“我觉得没关系”而保留所有分析。只有能证明某个具体分析没有受到改写影响时，才应保留它。

| Pass 做了什么 | Analysis 处理原则 |
|---|---|
| 只读、只打印/检查 IR | 可以保留所有 Analysis |
| 修改了 IR，且不确定影响范围 | 让默认失效机制处理 |
| 修改了 IR，但证明某项 Analysis 结果仍然正确 | 可以只保留那项 Analysis |

## 第五站：把缓存故事走一遍

想象一个小 pipeline：

```text
ReportConvCountPass → LowerConvPass → ReportConvCountPass
```

第一个 Pass 请求 `ConvCountAnalysis`，结果被计算并缓存。`LowerConvPass` 改动了 IR，因此相关分析不再保证正确。第二个报告 Pass 再请求时，不能盲用旧结果；AnalysisManager 会按失效规则重新计算。

如果中间的 Pass 完全没有修改 IR，并正确标记分析保留，那么结果可以继续使用。这样做既避免不必要的遍历，也保证缓存不会悄悄过期。

要点不是“尽可能缓存”，而是：

```text
缓存可以省计算；正确的失效规则保证缓存仍可信。
```

## 今天的练习路线

### 练习 A：把已有 walk 改成 Analysis

- 选一个 Module，写 `ConvCountAnalysis` 统计 `mynpu.conv` 数量。
- 写一个只读 Module Pass，通过 `getAnalysis<ConvCountAnalysis>()` 打印统计。
- 连续请求两次，理解 Analysis 的生命周期由管理器控制，不是每次都手写 `ConvCountAnalysis(op)`。

### 练习 B：观察失效

- 在统计 Pass 后运行一个会改变 IR 的 Pass。
- 再次请求统计 Analysis，确认它反映的是新 IR。
- 把中间 Pass 换成只读 Pass，并标记 Analysis 保留，解释两种情况下缓存结果是否还能用。

### 练习 C：写出不变量

为你的 Analysis 写一句话说明：

> “这个结果只在 ______ 没有发生变化时有效。”

例如 ConvCountAnalysis 的结果依赖当前 Operation 子树中的 Op 组成；新增或删除 Conv 都会使结果过期。

## 常见误会

### “Analysis 就是一个只读 Pass。”

相似但不是一回事。Pass 是 PassManager 调度的执行单元；Analysis 是 Pass 按需查询的计算结果，由 AnalysisManager 管理。

### “我请求两次，构造函数一定运行两次。”

不一定。结果有效时 AnalysisManager 会缓存并复用；被失效后再次请求才重新计算。

### “Analysis 里也可以顺手修 IR。”

不可以。Analysis 必须只读。需要改变 IR 的工作属于 Pass 或 Rewrite。

### “Pass 没调用 `markAllAnalysesPreserved()` 就表示它一定改了 IR。”

不是。它只是没有承诺分析仍有效，PassManager 会按保守规则处理。只有确实能证明时才主动声明保留。

### “每个 Pass 都应该自己缓存遍历结果。”

这会让失效和跨 Pass 复用难以管理。若信息是某个 Operation 当前 IR 的分析结果，优先考虑使用 MLIR Analysis 机制。

## 今天结束时，试着讲给别人听

1. Analysis 为什么不能修改 IR？
2. `getAnalysis<T>()` 第一次和后续请求时分别可能发生什么？
3. 为什么改写 IR 后缓存的 Analysis 可能过期？
4. 只读 Pass 为什么可以保留 Analysis？
5. Day 4 的 `walk` 与 AnalysisManager 各自解决什么问题？

讲清这些，就具备了理解更高级分析的基础。后面学习 Linalg、Bufferization 或模型优化时，你会反复看到 Pass 查询分析，再据此决定如何改写 IR。

## 参考资料

- [MLIR Pass Infrastructure：Analysis Management](https://mlir.llvm.org/docs/PassManagement/#analysis-management)
- [MLIR Pass Infrastructure：Preserving Analyses](https://mlir.llvm.org/docs/PassManagement/#preserving-analyses)
