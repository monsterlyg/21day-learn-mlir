# Day 12：Tensor、MemRef 和动态 Shape

## 从昨天的 SSA 值继续

昨天你看到 `%acc` 如何通过 BlockArgument 和 `scf.yield` 在循环中传递。今天继续问一个 AI 编译器很常见的问题：

> 这个 SSA Value 表示一个逻辑张量，还是一块可以读写的内存？它的维度现在已知吗？

今天的重点是区分 **tensor 值** 和 **memref 内存引用**，并看懂类型里的 `?`。这会为后面学习 Linalg 和 Bufferization 做准备；今天先不深入内存规划。

## 第一站：类型里的 `?` 表示什么？

```mlir
tensor<2x3xf32>
tensor<?x3xf32>
tensor<?x?xf32>
```

- `tensor<2x3xf32>`：rank 是 2，两个维度分别是 2 和 3。
- `tensor<?x3xf32>`：rank 仍是 2，第二维是 3，第一维的大小目前未知。
- `tensor<?x?xf32>`：rank 是 2，但两维大小都未知。

这里的 `?` 是**类型层面的动态维度**。它不表示张量没有 rank，也不表示一个“可以随便变形”的数组。它表示编译时不能把这一维固定成某个常量。

类型 `tensor<?x?xf32>` 本身也不携带“运行时第一维等于 7”这样的数值。运行时尺寸通常由 `index` 类型的 SSA Value 表示，并由函数参数或 shape 相关 Op 提供。

## 第二站：Tensor 是值，MemRef 是内存引用

先用一个日常类比：

- **Tensor** 像一份不可变的计算结果。你对它做变换，得到一个新的逻辑值；编译器之后可以决定如何实现它。
- **MemRef** 像指向一块内存的引用。你可以从指定位置读出元素，也可以把元素写回去。

看并排的两种写法：

```mlir
// Tensor：抽取元素，或产生一个逻辑上更新后的 tensor 值。
%value = tensor.extract %t[%i, %j] : tensor<?x?xf32>
%updated = tensor.insert %value into %t[%i, %j] : tensor<?x?xf32>

// MemRef：引用一个 buffer，并对它进行读取或写入。
%value2 = memref.load %buffer[%i, %j] : memref<?x?xf32>
memref.store %value2, %buffer[%i, %j] : memref<?x?xf32>
```

`tensor.insert` 返回新的 tensor SSA Value；`memref.store` 修改它所引用的内存。这个区别会影响优化：多个 tensor SSA 值可以表达函数式的数据流，而多个 memref 可能引用或别名同一块存储。

## 第三站：动态维度怎样在运行时提供？

假设函数接收两个运行时维度 `%m` 和 `%n`：

```mlir
func.func @make_tensor(%m: index, %n: index) {
  %t = tensor.empty(%m, %n) : tensor<?x?xf32>
  // 在这里继续构造或填充 tensor。
  return
}
```

这里：

- 类型 `tensor<?x?xf32>` 声明 rank 和元素类型；
- `%m`、`%n` 是运行时 dimension 值，类型为 `index`；
- `tensor.empty` 用这些动态尺寸创建一个具有对应 shape 的 tensor 值。

注意：`tensor.empty` **不表示元素已经被初始化成零**。它只提供一个指定 shape 的 tensor 值。要使用其中的元素，后续计算必须按语义写入或从其他定义良好的值构造内容。

同样的动态尺寸可以用于分配 memref：

```mlir
func.func @make_buffer(%m: index, %n: index) {
  %buffer = memref.alloc(%m, %n) : memref<?x?xf32>
  // 使用前先写入合法的数据。
  memref.dealloc %buffer : memref<?x?xf32>
  return
}
```

`memref.alloc` 分配内存，但不自动把内容清零。真实代码还要考虑分配和释放的生命周期；这里只是为了展示动态尺寸怎样进入操作数。

## 第四站：Shape inference 在回答什么问题？

Day 8 的 `mynpu.conv` 已实现 `InferTypeOpInterface`。它尝试根据 input、weight、stride 和 padding 推导 output 的类型和 shape。

对静态输入来说，公式直观：

```text
OH = (IH + 2 * PH - KH) / SH + 1
OW = (IW + 2 * PW - KW) / SW + 1
```

如果 `IH=224`、`KH=7`、`PH=3`、`SH=2`，那么：

```text
OH = (224 + 2*3 - 7) / 2 + 1 = 112
```

但如果 `IH` 在编译时是动态的，就不能拿 `?` 当普通整数代入公式。MLIR 中动态维度通常用 `ShapedType::kDynamic` 这样的特殊值表示。推导代码必须先检查每个参与计算的维度是否已知。

概念上可以这样想：

```cpp
if (inputHIsDynamic || kernelHIsDynamic)
  outputH = ShapedType::kDynamic;
else
  outputH = calculateOutputHeight(...);
```

Batch 维和输出通道也要传播：如果 input 的 N 是动态的，output 的 N 也应是动态的；如果 weight 的 OC 动态，output 的 OC 也应动态。

因此 shape inference 不是“尽可能算出数字”，而是**只推导当前信息足够确定的部分，其余维度诚实地保留为动态**。

## 回到 Day 8：检查当前 Conv 推导的边界

打开 `day8-interface/IR/MyNPUOps.cpp` 的 `ConvOp::inferReturnTypes`，对照这些问题读代码：

1. 它有没有检查 input 和 weight 的 rank 都是 4？
2. 它有没有检查 `stride` 的长度，以及 stride 是否大于 0？
3. 它怎样处理动态 `IH`、`IW`、`KH` 或 `KW`？
4. Batch 或 output channel 动态时，output type 是否保留动态维度？
5. padding/stride 不合法时，是返回 failure 还是继续算？

当前实现主要面向静态 shape。特别注意：若直接把动态 sentinel 当普通整数参与卷积公式，可能会算出错误结果。本日练习是让代码在遇到动态维度时保留 `kDynamic`，而不是假装得到了静态尺寸。

## 第五站：Shape 传播的小例子

设：

```text
input  : tensor<?x3x224x224xf32>
weight : tensor<64x3x7x7xf32>
stride : [2, 2]
padding: [3, 3]
```

input 的 batch 维未知，但空间维已知。正确的 output shape 应是：

```text
tensor<?x64x112x112xf32>
```

这正是“能推导的维度继续推导，未知的维度保留未知”。若 input 的高度也是 `?`，则 output 高度也应是 `?`，但宽度仍可以推导为 112。

## 今天的练习路线

### 练习 A：读类型

把下面类型各自的 rank、静态维度、动态维度写出来：

```text
tensor<8x?x16xf32>
memref<?x32xi8>
tensor<*xf32>
```

前两个是 ranked type，最后一个是 unranked tensor：连 rank 都未知。初学时优先使用 ranked type，因为它保留了更多有用信息。

### 练习 B：对比 tensor 和 memref

解释下面操作中，哪个产生新 SSA Value，哪个写入已有内存：

```mlir
%x = tensor.extract %t[%i] : tensor<?xf32>
%t2 = tensor.insert %x into %t[%i] : tensor<?xf32>
%y = memref.load %m[%i] : memref<?xf32>
memref.store %y, %m[%i] : memref<?xf32>
```

### 练习 C：补强 Conv shape inference

- 先处理 `inputH` 或 `inputW` 动态的情况。
- 再处理动态 batch 和动态 output channel 的传播。
- 检查 stride 是否为正，避免除以零或无效结果。
- 为每种情况写一个合法输入和一个预期失败/保留动态维度的输入。

## 常见误会

### “`?` 是一个运行时值。”

不是。`?` 是类型上的未知维度；运行时维度值由 `index` SSA Value 提供。

### “Tensor 和 MemRef 都是多维数组，只是拼法不同。”

它们的语义不同。Tensor 是值语义；MemRef 是对内存区域的引用，可以读写，也可能涉及别名。

### “`tensor.empty` 创建了全零 tensor。”

没有。它只提供 shape，不保证元素值。需要明确定义内容后才能依赖这些元素。

### “shape inference 必须把所有 `?` 都变成数字。”

不必，也不应该凭空猜测。信息不足时就保留动态维度。

### “Bufferization 就是把类型名字替换掉。”

不是。Tensor 到 MemRef 的变换还要决定结果写到哪里、能否复用输入 buffer、是否需要拷贝。完整算法留到 Day 15。

## 今天结束时，试着讲给别人听

1. `tensor<?x?xf32>` 的两个问号告诉我们什么？
2. 动态尺寸的运行时数值通常由什么类型的 SSA Value 表示？
3. Tensor 更新和 MemRef 写入的差异是什么？
4. Conv 某个输入维度未知时，shape inference 应该怎么处理对应输出维度？
5. 为什么 `tensor.empty` 的名字不能理解成“填满零”？

下一天我们会在这些基础上引入 Analysis：Pass 不改变 IR，而是先回答“这段 IR 有什么特征”，让后续决策能复用这些信息。

## 参考资料

- [MLIR Builtin Dialect：TensorType 与 MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/)
- [MLIR Tensor Dialect](https://mlir.llvm.org/docs/Dialects/TensorOps/)
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
