# Day 14：Linalg——把张量计算写成可分析的循环

## 今天要解决什么问题？

Day 12 你学会了区分 `tensor` 和 `memref`，Day 13 你学会了让 Pass 通过 Analysis 读取 IR。现在换一个问题：

> 一个「逐元素加法」或者「矩阵乘」，怎样写成 IR，才能让编译器知道谁和谁相乘、哪些维度可以并行、哪些维度必须累加？

如果只写 `scf.for` 和 `memref.load/store`，编译器能看到循环，却很难看出「这是一个矩阵乘」。Linalg 的价值就在这里：它用一份**结构化描述**同时表达计算和循环结构，让后续的融合、分块、向量化、lowering 都有依据。

今天的目标：

1. 读懂 `linalg.generic` 的 `indexing_maps`、`iterator_types` 和 region。
2. 从 named op（`linalg.add`、`linalg.matmul`）反推出 generic 形式。
3. 观察 `linalg-fuse-elementwise-ops` 如何把两个 op 合成一个。

## 环境

本文命令使用本机安装的 MLIR：

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
```

所有例子都可以直接用 heredoc 喂给 `$MLIR_OPT`，不需要额外文件。

## 第一站：named op 是 generic 的简写

先写一个最普通的逐元素加法：

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

运行：

```bash
$MLIR_OPT --linalg-generalize-named-ops <<'EOF'
module {
  func.func @add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %init = tensor.empty() : tensor<2x3xf32>
    %r = linalg.add ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>) outs(%init : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
EOF
```

输出：

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>)
        outs(%0 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
```

可以看到，`linalg.add` 只是 `linalg.generic` 的一种固定写法。一个 `linalg.generic` 由三部分组成：

- **operands / results**：`ins` 是只读输入，`outs` 是输出初始化，结果类型由 `->` 给出。
- **indexing_maps**：每个 operand/result 各自的「循环维到张量维」映射。
- **iterator_types + region**：循环维是并行还是归约，以及每个元素怎么算。

`ins/outs` 这种写法叫 **Destination Passing Style（DPS）**。即使是 tensor 语义，也要显式给出 `outs`，告诉 Linalg「结果写到哪里、初值是什么」。

## 第二站：indexing map 在描述什么？

先看这一行：

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
```

左边 `(d0, d1)` 是**迭代空间**（iteration space），右边是**张量下标**。对 `tensor<2x3xf32>` 来说：

```text
d0 ∈ [0, 2)
d1 ∈ [0, 3)
```

`(d0, d1) -> (d0, d1)` 表示「迭代空间里的 (d0, d1) 直接对应张量的 (d0, d1)」。

广播则是省略某些维：

```mlir
#id    = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d1)>
```

`tensor<3xf32>` 没有 `d0` 这一维，所以用 `#bcast`。完整的广播加法：

```mlir
#id = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d1)>
module {
  func.func @bcast_add(%a: tensor<2x3xf32>, %b: tensor<3xf32>) -> tensor<2x3xf32> {
    %init = tensor.empty() : tensor<2x3xf32>
    %r = linalg.generic {indexing_maps = [#id, #bcast, #id],
                         iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<2x3xf32>, tensor<3xf32>)
        outs(%init : tensor<2x3xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %s = arith.addf %in, %in_0 : f32
        linalg.yield %s : f32
    } -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
```

这段可以直接用 `$MLIR_OPT` 解析并打印，说明 map 的数量和 operand 数量必须一一对应。

## 第三站：reduction 在哪里？

再看矩阵乘：

```mlir
%r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
                   outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
```

`--linalg-generalize-named-ops` 会把它展开成：

```mlir
#map  = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
%r = linalg.generic {indexing_maps = [#map, #map1, #map2],
                     iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
    outs(%init : tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %m = arith.mulf %in, %in_0 : f32
    %s = arith.addf %out, %m : f32
    linalg.yield %s : f32
}
```

现在三个维度各有分工：

```text
d0：输出行，parallel
d1：输出列，parallel
d2：K 维，reduction
```

等价的伪代码是：

```text
for d0 in [0, 2):
  for d1 in [0, 4):
    for d2 in [0, 3):
      out[d0, d1] += a[d0, d2] * b[d2, d1]
```

注意 region 里读的是 `%out`，不是忽略它。对 reduction 来说，`outs` 提供累加初值；如果初值未定义，结果就没有意义。这就是为什么 `tensor.empty` 不能随便当成「零初始化」。

## 第四站：elementwise fusion 做了什么事？

准备一个「先加、再平方」的两级计算：

```mlir
module {
  func.func @fused(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
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

运行（把上面的输入保存为 `fused.mlir`）：

```bash
$MLIR_OPT -p 'builtin.module(linalg-generalize-named-ops,linalg-fuse-elementwise-ops)' fused.mlir
```

输出会变成一个 generic：

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fused(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map],
                         iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>)
        outs(%0 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.mulf %2, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
```

观察三件事：

1. 两个 op 变成一个，中间的 `tensor.empty` 消失了。
2. 两个 region 被串进同一个 region，`%2` 直接喂给 `%3`。
3. 迭代空间必须兼容：两个 op 都是 `(d0, d1) -> (d0, d1)`、都是 parallel，才能逐元素融合。

这也是为什么它叫 **elementwise** fusion。`linalg.matmul` 带 reduction，不会用这条规则和普通逐元素 op 直接融合；那属于 tiling/fusion 的另一类问题。

## 第五站：named 和 generic 的往返

两个方向的命令：

```bash
# named → generic
$MLIR_OPT --linalg-generalize-named-ops input.mlir

# generic → named（能识别时）
$MLIR_OPT --linalg-specialize-generic-ops input.mlir
```

为什么两种形式都保留？

- **named op** 表达意图更清楚，报错更友好，某些 lowering 直接匹配 named op。
- **generic** 是统一形式，融合、分析、通用 lowering 只需要处理一种结构。

一个实用习惯：遇到不认识的 named op，先 generalize，再读 maps 和 iterator types。理解了 generic，named op 就只是「预设好的 generic」。

## 今天的练习路线

### 练习 A：读 matmul

不看答案，回答：

1. 三个 map 分别对应哪个 operand？
2. `iterator_types` 为什么是 `parallel, parallel, reduction`？
3. 如果 `a` 是 `tensor<2x3xf32>`、`b` 是 `tensor<3x4xf32>`，每个循环维的边界是多少？

### 练习 B：写一个广播加法

把 `tensor<2x3xf32>` 和 `tensor<3xf32>` 相加，要求：

1. 用 `linalg.generic` 手写。
2. 用 `$MLIR_OPT` 确认能解析。
3. 再用 `--linalg-specialize-generic-ops` 看它是否能变回 named op。

### 练习 C：预测 fusion

把练习 B 的结果接到一个 `linalg.mul` 后面，先写下你预期的输出，再运行 `linalg-fuse-elementwise-ops` 对照。

### 练习 D：负例

把 `linalg.matmul` 和一个逐元素 `linalg.add` 串起来，运行：

```bash
$MLIR_OPT -p 'builtin.module(linalg-generalize-named-ops,linalg-fuse-elementwise-ops)' matmul_add.mlir
```

观察 matmul 是否被融合，并解释原因。这里的「没有变化」和「报错」一样重要：它说明规则有适用边界。

## 常见坑

- **map 数量对不上**：`indexing_maps` 的数量必须等于 `ins` + `outs` 的数量。
- **iterator_types 长度对不上**：它的长度等于 map 左边的迭代维数（例如 `(d0, d1)` 就是 2）。
- **把 `outs` 当成无关参数**：reduction 依赖它的初值；elementwise 也应该给它一个合法的结果 buffer。
- **tensor 和 memref 混用**：本文都在 tensor 上操作；bufferize 之后同样的 linalg op 会变成 memref 版本，语义是原地写。
- **以为 fusion 一定发生**：多用户、不同迭代空间、非 elementwise 都可能阻止融合。

## 今天结束时，试着讲给别人听

1. `linalg.generic` 的三要素分别回答什么问题？
2. `(d0, d1) -> (d1)` 为什么表示广播？
3. matmul 的 `reduction` 维在循环里对应哪一层？
4. elementwise fusion 需要满足什么条件？
5. 为什么有了 named op 还需要 generic？

## 参考资料

- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Linalg Rationale](https://mlir.llvm.org/docs/Rationale/LinalgRationale/)
- 本机源码：`mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td`、`mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td`
- 本机测试：`mlir/test/Dialect/Linalg/`、`mlir/test/Dialect/Linalg/fusion-elementwise-ops.mlir`
