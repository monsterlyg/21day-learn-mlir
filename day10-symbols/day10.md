# Day 10：函数名不是 SSA 值——理解 Symbol 和调用

## 今天要解开的疑问

昨天我们整理的是一个 Op 里面的计算。今天把视线拉远一点：一个 MLIR Module 里可能有多个函数，函数之间会互相调用。

看到下面这行时：

```mlir
%r = func.call @twice(%x) : (i32) -> i32
```

你可能会问：`%x` 和 `@twice` 看起来都像“引用”，它们是不是同一种东西？今天的答案是：**不是。`%x` 是 SSA Value，`@twice` 是 Symbol 引用。**

## 第一步：先读一个完整的小 Module

```mlir
module {
  func.func private @twice(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %x, %c2 : i32
    return %r : i32
  }

  func.func @main(%input: i32) -> i32 {
    %result = func.call @twice(%input) : (i32) -> i32
    return %result : i32
  }
}
```

沿着一行调用逐个看：

- `%input` 是调用的实参，是一个 SSA Value，有明确的类型 `i32`。
- `@twice` 是被调用函数的名字，写在 SymbolRef 位置。
- `%result` 是 `func.call` 的返回 SSA Value。
- `@twice` 的定义是另一个 `func.func` Operation。

如果把 `%input` 想成“这次计算传过来的数据”，那么 `@twice` 更像“去符号目录里查这个名字代表哪个函数”。

## 第二步：为什么不能把函数名当 SSA Value？

SSA Value 由某个 Operation 或 BlockArgument 产生，能直接连到它的使用者。它有定义位置、类型和 use-def 链。

Symbol 则是一个命名的 Operation。它通过 SymbolTable 中的名字被查找。SymbolRef 不会像 `%value` 那样成为普通 SSA use，也不会作为计算数据传给算术 Op。

| 对比 | SSA Value，例如 `%input` | Symbol 引用，例如 `@twice` |
|---|---|---|
| 指向什么 | 某次计算的结果或块参数 | 命名的 Operation，常见是函数或全局量 |
| 怎么写 | `%` 开头 | `@` 开头 |
| 主要作用 | 在计算中传递数据 | 跨作用域按名字寻找定义 |
| 常见使用 | `arith.addi %a, %b` | `func.call @foo(%a)` |
| 典型查找 | SSA use-def 链 | SymbolTable / SymbolRefAttr |

函数调用里两种引用同时出现：函数名是 SymbolRef，实参和结果是 SSA Values。

## 第三步：谁负责保存这些名字？

在上面的例子里，两个函数都位于 `module` 里面。Module 是一个 SymbolTable：它负责维护里面的符号名，并要求同一张表中的符号名不冲突。

```text
module（SymbolTable）
├── func.func @twice（Symbol）
└── func.func @main（Symbol）
       └── func.call @twice（SymbolRef）
```

`func.func` 定义符号；`func.call` 保存对符号的引用。解析 IR 时，MLIR 能检查这个调用在合适的符号表中是否能找到目标，并检查调用签名是否匹配。

SymbolTable 也可以嵌套。比如一个符号表中有一个命名的子模块，子模块里又有函数，引用可以写成多段路径：

```mlir
@package::@helper
```

初学阶段先理解最常见的 `@function` 就够了；嵌套路径留作阅读拓展。

## 第四步：做一个小观察练习

在上面的 Module 上，用 MLIR parser/打印工具做 round-trip：读入后再打印，观察 `%` 和 `@` 各自保留在哪里。然后回答：

1. `@twice` 是由哪一个 Op 定义的？
2. 哪个 Op 引用了它？
3. 调用的输入和输出分别是什么 SSA Value？
4. 如果改成 `func.call @missing(...)`，为什么这不是“找不到一个 SSA 定义”的问题？

接着把 `@twice` 改名为 `@scale2`。要一起改函数定义和 `func.call` 的引用；只改其中一个会造成悬空引用。

## 第五步：C++ 里如何查找 Symbol？

当你在 Pass 中拿到一个引用符号的 Operation，可以从最近的 SymbolTable 开始查找。示意代码如下：

```cpp
mlir::SymbolRefAttr callee = callOp.getCalleeAttr();
mlir::Operation *target =
    mlir::SymbolTable::lookupNearestSymbolFrom(callOp, callee);
```

如果需要更完整的访问/修改支持，MLIR 提供 `SymbolTable` 的查找和符号使用枚举 API；引用符号的自定义 Op 还可以实现 `SymbolUserOpInterface`。

初学阶段最重要的是知道这不是 SSA 的 `getDefiningOp()`：

- 对 `%value`，沿 SSA use-def 找它的定义；
- 对 `@name`，在正确的 SymbolTable 作用域里按名称解析。

## 常见误会

### “`func.call` 的 `@foo` 是普通 Attribute，所以没有规则。”

文本上它确实由 SymbolRefAttr 表示，但它有符号解析和引用语义。改名、移动或删除符号时，要考虑引用是否仍有效。

### “`@foo` 能直接传给 `arith.addi`。”

函数名不是整数或张量 SSA Value。若要把一个符号表示的全局对象变成可计算的地址/值，方言通常需要专门的 materialize/address-of Op。

### “名字唯一就够了。”

还要在正确作用域内解析。嵌套 SymbolTable 会影响从哪里开始查找，符号 visibility 也会影响跨模块访问规则。

## 今天的练习路线

### 必做

- 读懂 `@twice` 和 `@main` 的示例。
- 用自己的话解释 `%input` 与 `@twice` 的区别。
- 修改函数名并同步调用点；再故意只修改定义，观察 verifier/解析工具的诊断。
- 在纸上画出 Module SymbolTable、两个函数、一次调用，以及调用中的 SSA Value。

### 选做

- 在当前 MLIR 文档中阅读 `SymbolRefAttr` 和 `SymbolTable::lookupNearestSymbolFrom`。
- 阅读一个有嵌套 `@module::@function` 的例子，解释每一段名字对应哪层 SymbolTable。
- 看 `func.call` 是否能引用声明但没有函数体的外部函数，并说明这在什么场景有用。

## 今天结束时，你应该能讲清

1. Module 为什么是 SymbolTable？
2. 函数定义和函数调用分别如何参与 Symbol/引用关系？
3. SSA use-def 链和 SymbolRef 的名字查找有什么区别？
4. 改名或删除一个函数时，为什么要检查它的 symbol uses？

讲清楚这些之后，下一天我们进入函数体内部：Region 和 Block 如何表达分支、循环，以及值如何通过 `scf.yield` 从一段控制流返回。

## 参考资料

- [MLIR Symbols and Symbol Tables](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)
- [MLIR Func Dialect](https://mlir.llvm.org/docs/Dialects/Func/)
