# Day 15：Bufferization——从 tensor 值走向存储

## 今天的目标

Bufferization 将 tensor 风格的数据流转换为可实现的内存读写。它不是单纯改类型：它需要决定是否能复用已有 buffer、是否存在别名冲突、是否需要分配和复制。今天先观察 One-Shot Bufferize，不深入内存规划或自定义接口。

## 1. 为什么原地写入不总是安全

考虑 `%next = tensor.insert %v into %t[%i]`。tensor 语义下 `%next` 是新逻辑值，旧 `%t` 仍可被其他使用者观察。若 bufferization 直接写回 `%t`，而旧值之后还要读取，语义就会改变。因此它必须分析读写、use-def、别名与冲突；必要时分配新 buffer 或复制。

是否能原地实现，取决于旧值是否仍活跃、其他值是否可能别名同一存储，以及该 Op 的 bufferization 模型。

## 2. One-Shot Bufferize 的观察目标

概念流程：

```text
tensor IR → 分析读写与冲突 → 选择复用或新存储 → memref 相关 IR
```

转换后关注 `memref.alloc`、copy、load/store、函数边界，以及是否有 tensor Op 留下。Bufferization 完成并不等于生成机器码，后面仍可能要降低 memref、SCF、arith 等方言。

## 3. 本机实验

先查实际支持项：

```sh
mlir-opt --help | rg -i 'bufferiz|one-shot|linalg'
```

从当前版本测试中找一个合法的 Linalg/tensor 示例，再按帮助中的 pass 名称运行。`--one-shot-bufferize` 只是常见拼法，不保证每个版本都提供完全相同选项；遇到未注册时以本机 help 和测试为准。

保存变换前后 IR，并回答：

- 哪些 tensor 值变成了 memref？
- 哪里出现分配或复制？分别对应哪些旧值仍然被使用？
- 哪些操作仍是 tensor？是否说明 pipeline 尚未处理它们？
- 哪些结果可能复用输入存储？证据是什么？

## 4. 动手路线

### A. 追踪 use-def

用 Day 14 的 elementwise 示例。对关键 tensor 值列出读取者、更新结果以及旧值后续是否继续使用，然后对应观察到的 alloc/copy。

### B. 制造冲突

让更新前的 `%t` 在生成 `%next` 后仍被另一个计算使用。重新 bufferize，比较结果。解释复制如何维持值语义；若没有复制，继续查工具的别名/读写模型。

### C. 看函数边界

比较带 tensor 参数/返回值的函数与内部计算，记录签名、调用点如何变化。行为依赖版本、选项和 pipeline，必须以实际 IR 为证。

## 常见误会

- Bufferization 需要维护值语义，不只是换类型。
- Copy 可能是保持旧值正确所必需，不应仅凭“有复制”判定优化失败。
- 没有 copy 也要确认 Op 的别名与读写模型正确。
- bufferize 后通常还需要后续 lowering。

## 自测

1. 旧 tensor 仍被使用时，原地写入有什么风险？
2. One-Shot Bufferize 输出的 copy 可能保护什么语义？
3. 转换后为何还没有得到机器码？

## 参考资料

- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
