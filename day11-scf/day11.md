# Day 11：走进 Region——用 `scf.if` 和 `scf.for` 表达控制流

## 从昨天的函数继续往里走

昨天我们看了一个 Module 里的多个函数，以及函数之间如何通过 `@name` 调用。今天选中一个函数，进入它的函数体：如果输入满足条件就走一条路，否则走另一条；或者重复执行一段计算，直到循环结束。

在普通文字里我们会写 `if` 和 `for`。在 MLIR 中，这些控制流可以由嵌套的 Region、Block、BlockArgument 和 terminator 来表达。今天先学习结构清楚的 `scf` 方言，不急着写底层的 branch。

## 先把几个名字翻译成人话

- **Region**：一个 Op 内部装着的一段 IR。可以把它想成“Op 里面的小程序”。
- **Block**：Region 中的一段基本执行单元，里面按顺序放 Operations。
- **BlockArgument**：Block 的输入参数，像一个小函数的形参。
- **terminator**：结束当前 Block 的 Op；它可能还会把值交给外层或下一段控制流。

一个 `scf.if` 有 then Region，也可能有 else Region；一个 `scf.for` 有循环体 Region。Region 让控制结构保持嵌套、可读，而不是一开始就摊成跳转箭头。

## 第一站：`scf.if` 不只是“选一个分支”

看这个例子：

```mlir
func.func @choose(%cond: i1, %left: i32, %right: i32) -> i32 {
  %selected = scf.if %cond -> (i32) {
    scf.yield %left : i32
  } else {
    scf.yield %right : i32
  }
  return %selected : i32
}
```

读它时按这条线走：

1. `%cond` 是 `i1`，决定选 then 还是 else。
2. then Region 用 `scf.yield %left` 把结果交出来。
3. else Region 用 `scf.yield %right` 交出另一个结果。
4. 两个分支都必须交出和 `-> (i32)` 对应的结果。
5. 外层得到 `%selected`，继续像普通 SSA Value 一样使用。

**Region 是控制流，`scf.yield` 是它的出口，`%selected` 是控制流 Op 的结果。** 把这三者连起来，很多嵌套 IR 就容易读了。

如果 `scf.if` 没有结果，分支里通常不需要传值；今天重点看有结果的版本，因为它最能展示 Region 如何把值交回去。

## 第二站：`scf.for` 里的循环变量去哪了？

下面的循环把 0、1、2、3 加起来，最后得到 6：

```mlir
func.func @sum_to_four() -> i32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32

  %sum = scf.for %i = %c0 to %c4 step %c1
      iter_args(%acc = %zero) -> (i32) {
    %i32 = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %i32 : i32
    scf.yield %next : i32
  }

  return %sum : i32
}
```

把它拆成四个角色：

| 名称 | 作用 |
|---|---|
| `%c0` | 循环下界，包含在循环范围内 |
| `%c4` | 循环上界，不包含在循环范围内 |
| `%c1` | 每次递增的步长，必须为正 |
| `%i` | 循环体 Region 的 BlockArgument，也就是 induction variable |
| `%zero` | 累加值的初始 SSA Value |
| `%acc` | 循环体中的 BlockArgument，表示本轮开始时的累加值 |
| `%next` | 本轮算出的新累加值 |
| `scf.yield %next` | 把新累加值传给下一轮；最后一轮的值成为 `%sum` |

可以把 `iter_args` 想成“循环携带的变量”：进入循环时拿 `%zero` 初始化，循环体每次用 `%acc` 读当前值，再用 `scf.yield` 交出下一个值。

`scf.for` 的范围是下界包含、上界不包含。因此这里 `%i` 取 `0, 1, 2, 3`，不是 `0` 到 `4` 共五次。

## 第三站：画出 BlockArgument 和 SSA 流向

把循环画成这样：

```text
外层：%zero ── 初始值
                │
                ▼
循环体 Region：(%i, %acc)  ← BlockArguments
                    │
                    ├── %next = %acc + %i
                    │
                    └── scf.yield %next
                              │
                 下一轮的 %acc / 最终的 %sum
```

注意 `%acc` 不是每轮新建的全局变量。它是循环体 Block 的参数；`scf.yield` 把值交回 `scf.for`，由外层循环语义决定它在下一轮如何绑定。

这和函数调用有一点相似：外层把值交给内部区域，内部区域通过自己的参数名接收；区别是循环会重复执行，并且 BlockArgument 的值可能随轮次改变。

## 第四站：自己读一段嵌套控制流

把前面的例子组合起来：循环每轮判断 `%i` 是否为偶数，偶数加到总和里，奇数保持原总和。

```mlir
func.func @sum_even_below_four() -> i32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %two = arith.constant 2 : i32
  %sum = scf.for %i = %c0 to %c4 step %c1
      iter_args(%acc = %zero) -> (i32) {
    %i32 = arith.index_cast %i : index to i32
    %rem = arith.remsi %i32, %two : i32
    %is_even = arith.cmpi eq, %rem, %zero : i32
    %next = scf.if %is_even -> (i32) {
      %added = arith.addi %acc, %i32 : i32
      scf.yield %added : i32
    } else {
      scf.yield %acc : i32
    }
    scf.yield %next : i32
  }

  return %sum : i32
}
```

读嵌套 IR 时不要一次试图理解所有括号。按层次读：

1. 外层 `scf.for` 管理循环次数和累加值。
2. 循环体先计算 `%is_even`。
3. 内层 `scf.if` 选出 `%next`。
4. 外层 `scf.yield` 把 `%next` 交回循环。

如果迷路，就先找每个 Region 的 terminator：`scf.yield` 告诉你这段 Region 往外传什么。

## 今天的练习路线

### 练习 A：分支结果

- 将 `%left` / `%right` 的类型改成 `f32`，并同步修改 `scf.if` 的结果类型。
- 故意让一个分支 yield `i32`、另一个 yield `f32`，观察 verifier 为什么拒绝。
- 去掉 `else` 后思考：这个 `scf.if` 是否还能产生一个必定有值的结果？

### 练习 B：循环携带值

- 将循环上界改成 5，手工列出 `%i` 每轮的值。
- 预测 `@sum_to_four` 的返回值，再通过解释器/JIT 或在纸上逐轮计算确认。
- 把 `%acc` 改成两个累加值，例如同时求和和计数，观察 `iter_args`、Region 参数、`scf.yield` 和 `scf.for` 结果如何成组对应。

### 练习 C：画 IR 结构

对 `@sum_even_below_four` 画出：

- 外层函数 Region；
- `scf.for` 的循环体 Region 和 BlockArguments；
- 内层 `scf.if` 的 then/else Regions；
- 每条 `scf.yield` 传出的值。

## 常见卡点

### `scf.yield` 是不是函数返回？

不是。它结束当前 SCF Region，并按所在 Op 的规则传递值。函数返回由 `return` 表达。

### `%i` 是在循环前定义的吗？

不是。`%i` 是循环体 Region 的 BlockArgument，由 `scf.for` 每轮提供。

### `%acc` 为什么既像输入又像输出？

进入循环时，初始值绑定到 `%acc`；结束一轮时，`scf.yield` 把更新值交回循环。静态类型保持相同，运行时值逐轮变化。

### Region 就等于 Block 吗？

不一定。一个 Region 可以含一个或多个 Block。今天的 SCF 例子通常每个相关 Region 使用单个 Block，但其他方言可能表达更复杂的控制流。

## 今天结束时，试着讲给别人听

沿着 `@sum_to_four` 解释一遍：

1. 循环体为什么需要 `%i` 和 `%acc` 两个 BlockArgument？
2. `iter_args` 如何把 `%zero` 和 `%acc` 连起来？
3. `scf.yield` 为什么既影响下一轮，也决定最后的 `%sum`？
4. `scf.if` 的结果和普通 SSA Value 有什么关系？

下一天我们从“值怎样流过控制流”转向“张量值和内存引用有什么不同”，并用动态 shape 看 MLIR 类型里哪些尺寸已知、哪些要到运行时才知道。

## 参考资料

- [MLIR SCF Dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
