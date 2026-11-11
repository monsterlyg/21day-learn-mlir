# Day 15：Bufferization——什么时候必须复制

## 今天要解决什么问题？

Day 14 里的 Linalg 都跑在 `tensor` 上。tensor 是**值**：你读它、产生新的 tensor，原来的值不变。但 CPU/GPU 最终要操作的是内存，函数调用也要传递 buffer。于是出现一个必须回答的问题：

> tensor 什么时候变成 memref？新结果是复用输入的内存，还是另外分配一块？如果复用，会不会踩到还在被读的数据？

MLIR 的 One-Shot Bufferize 就是在回答这个问题。它先做别名和冲突分析，再决定每个 `tensor.empty` 变成 `memref.alloc`、复用已有 buffer，还是插入一次 copy。

今天的目标：

1. 看懂 bufferize 前后的函数签名和 `memref.alloc`。
2. 理解为什么有些结果能复用、有些必须新分配。
3. 知道 `memref.dealloc` 何时出现，以及「buffer 逃逸」是什么意思。

## 环境

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
```

## 第一站：tensor 和 memref 的桥

在完整 bufferize 之前，MLIR 允许显式跨越边界：

```mlir
func.func @bridge(%t: tensor<4xf32>) -> tensor<4xf32> {
  %b = bufferization.to_buffer %t : tensor<4xf32> to memref<4xf32>
  %r = bufferization.to_tensor %b : memref<4xf32> to tensor<4xf32>
  return %r : tensor<4xf32>
}
```

- `bufferization.to_buffer`：把 tensor 值变成可读写的 memref。
- `bufferization.to_tensor`：把 memref 重新看成 tensor 值。

这两个 op 不是免费的语义转换，它们表达的是「这里有一个 buffer 视图」。真实的 bufferize 会尽量消除它们。

注意：`to_tensor` 默认并不保证 memref 不会被别人继续写。若你能证明没有别名，可以用 `restrict` 等修饰；不能证明时，后续可能需要 copy。

## 第二站：跑一遍 One-Shot Bufferize

输入：

```mlir
module {
  func.func @add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %init = tensor.empty() : tensor<2x3xf32>
    %r = linalg.add ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
                       outs(%init : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
```

命令（把上面的输入保存为 `input.mlir`）：

```bash
$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1})' input.mlir
```

输出：

```mlir
module {
  func.func @add(%arg0: memref<2x3xf32, strided<[?, ?], offset: ?>>,
                 %arg1: memref<2x3xf32, strided<[?, ?], offset: ?>>)
      -> memref<2x3xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    linalg.add ins(%arg0, %arg1 : memref<2x3xf32, strided<[?, ?], offset: ?>>,
                    memref<2x3xf32, strided<[?, ?], offset: ?>>)
               outs(%alloc : memref<2x3xf32>)
    %cast = memref.cast %alloc : memref<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    return %alloc : memref<2x3xf32>
  }
}
```

先观察四个变化：

1. 函数参数从 tensor 变成 memref。
2. `tensor.empty` 变成 `memref.alloc`。
3. `linalg.add` 的结果从 SSA 值变成「写到 `outs` 指定的 buffer」。
4. 返回类型从 tensor 变成 memref，调用者拿到一块 buffer。

`alignment = 64` 是 bufferize 的默认对齐设置，不是语言要求。

## 第三站：为什么参数类型这么长？

`memref<2x3xf32, strided<[?, ?], offset: ?>>` 看起来比 `memref<2x3xf32>` 复杂。原因是函数边界上的 buffer 可能来自外部，外部不保证是紧凑连续布局，所以 bufferize 用**动态 stride** 描述它。

如果你希望边界使用 identity layout，可以显式指定：

```bash
$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map})' input.mlir
```

输出会变成：

```mlir
func.func @add(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32>
```

这是 ABI 设计问题，不是对错问题：动态 stride 更通用，identity layout 更简单、更利于后续 lowering。

## 第四站：什么时候复用，什么时候新分配？

把「先加、再平方」接起来：

```mlir
module {
  func.func @chain(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %i0 = tensor.empty() : tensor<2x3xf32>
    %s = linalg.add ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
                       outs(%i0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %i1 = tensor.empty() : tensor<2x3xf32>
    %r = linalg.mul ins(%s, %s : tensor<2x3xf32>, tensor<2x3xf32>)
                       outs(%i1 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
```

bufferize 后仍然是两块 `memref.alloc`：

```mlir
%alloc = memref.alloc() : memref<2x3xf32>
linalg.add ... outs(%alloc)
%alloc_0 = memref.alloc() : memref<2x3xf32>
linalg.mul ... outs(%alloc_0)
```

为什么第二个 op 不直接写回 `%alloc`？注意输入 IR 里 `%i1` 是一个独立的 `tensor.empty`，而 `%s` 又被 `linalg.mul` 作为输入读取。要让结果复用 `%alloc`，需要同时满足：op 支持原地写、`outs` 的旧值不被需要、没有其他冲突的读者/写者，并且分析能证明读写顺序安全。One-Shot Bufferize 在证明不了时会选择为新 `outs` 分配 buffer。

这里没有插入 `memref.copy`，因为新 buffer 本来就允许直接写；真正会出现 copy 的场景是「必须把旧内容搬到新位置」或「tensor 有别名、写之前要先保住旧值」。

可以用一句日常话记住：

```text
能证明安全 → 复用；
证明不了，但可以另开一块 → 新分配；
连另开一块都不满足语义（例如必须保留旧值/返回别名）→ copy。
```

## 第五站：`memref.dealloc` 什么时候出现？

把函数改成不返回结果：

```mlir
module {
  func.func @consume(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) {
    %init = tensor.empty() : tensor<2x3xf32>
    %r = linalg.add ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
                       outs(%init : tensor<2x3xf32>) -> tensor<2x3xf32>
    return
  }
}
```

运行：

```bash
$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1},buffer-deallocation-pipeline)' consume.mlir
```

输出中会出现：

```mlir
%alloc = memref.alloc() : memref<2x3xf32>
linalg.add ... outs(%alloc)
memref.dealloc %alloc : memref<2x3xf32>
```

对比上一站：当 buffer 作为函数返回值逃逸时，所有权交给调用者，函数内不能 `dealloc`；当 buffer 在函数内结束生命周期时，deallocation pipeline 才会补上释放。

这也解释了为什么「bufferize 完就结束」通常不是最终形态：你还欠一个所有权处理步骤。

## 第六站：bufferize 之后继续 lowering

一旦有了 memref，Linalg 就可以降到循环：

```bash
$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1},convert-linalg-to-loops)' input.mlir
```

会得到嵌套的 `scf.for` + `memref.load/store`：

```mlir
%alloc = memref.alloc() : memref<2x3xf32>
scf.for %i = %c0 to %c2 step %c1 {
  scf.for %j = %c0 to %c3 step %c1 {
    %0 = memref.load %arg0[%i, %j] : memref<2x3xf32, ...>
    %1 = memref.load %arg1[%i, %j] : memref<2x3xf32, ...>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %alloc[%i, %j] : memref<2x3xf32>
  }
}
```

这一步的意义是：从这里开始，IR 已经是「可以继续降到 LLVM」的普通控制流和内存访问。Linalg 的结构化信息在更低层不再存在。

## 今天的练习路线

### 练习 A：数分配

对 `chain` 例子分别运行：

1. `one-shot-bufferize` 单独；
2. `one-shot-bufferize + buffer-deallocation-pipeline`；
3. `one-shot-bufferize + convert-linalg-to-loops`。

记录每一步的 `memref.alloc` 数量和 `memref.dealloc` 数量，并解释变化。

### 练习 B：让 buffer 逃逸

把 `consume` 改成返回 `%r`，重新跑 deallocation pipeline，观察 `memref.dealloc` 是否消失。用「所有权」解释结果。

### 练习 C：改边界布局

分别用默认设置和 `identity-layout-map` bufferize，比较函数签名。想一想：如果这个函数要和 C 代码交互，你更愿意选哪种？

### 练习 D：负例

构造一个函数，让同一个 tensor 在两个地方被写：

```mlir
%a = linalg.add ... outs(%init)
%b = linalg.mul ... outs(%init)   // 同一个 init 被两个 op 写
```

先预测 bufferize 会复用还是新分配，再运行验证。这个练习的重点不是猜对，而是用「读写冲突」解释结果。

## 常见坑

- **不加 `bufferize-function-boundaries`**：函数参数和返回值可能仍是 tensor，边界不完整。
- **以为 `memref.alloc` 会清零**：它只分配，不初始化。
- **在 `tensor.empty` 上直接读**：没有定义的良好内容，bufferize 不会替你补。
- **看到 `memref.cast` 就慌**：它通常只是布局适配，不一定是 copy。
- **忘记 deallocation**：分配了却没有释放，或者在 buffer 逃逸时错误地释放。
- **把 bufferize 当成优化**：它首先是一次语义/表示转换，性能好不好取决于后续优化。

## 今天结束时，试着讲给别人听

1. tensor 和 memref 的语义差别是什么？
2. bufferize 为什么要做读写冲突分析？
3. 什么情况下会出现 `memref.copy`？
4. 为什么返回的 buffer 不能在函数内 `dealloc`？
5. 为什么 bufferize 之后还要继续 lowering？

## 参考资料

- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/#one-shot-bufferize)
- 本机源码：`mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp`、`OneShotModuleBufferize.cpp`
- 本机测试：`mlir/test/Dialect/Bufferization/Transforms/one-shot-bufferize*.mlir`
