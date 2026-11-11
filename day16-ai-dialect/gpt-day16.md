# Day 16：AI 高层 Dialect——TOSA 与 TPU-MLIR Top

## 今天的目标

结构化 Linalg 表达计算过程；AI 编译器还常保留 Conv、激活、布局和量化等模型级语义。今天选择一个高层 Dialect，围绕一个 Conv 或 elementwise Op 阅读定义、约束和 lowering。若学习目标是 TPU-MLIR，优先跟当前 checkout 的 Top 表达与 pipeline，TOSA 作对照阅读。

## 1. 先确认手头的源码和工具

在仓库中搜索 Op 定义、TOSA 文件或 lowering 入口。若当前工作区没有相关源码或工具，就阅读对应版本的公开定义，并明确记录未能本机运行；不要把概念示例说成已验证的 pipeline。

## 2. 一张 Op 说明卡

选择 Conv 或 Add，按顺序记录：

1. Op 名、operands、results、属性及类型；
2. verifier 检查的条件；
3. shape/layout、stride/padding 或广播语义；
4. 量化信息在哪里表示；
5. 哪个 pass 将它转换到什么表示；
6. 若可运行，保存变换前后 IR 和精确命令。

对 TOSA 可查看 `tosa.conv2d` 或 `tosa.add`；对 TPU-MLIR Top 则依当前项目的 Op 类、属性和 pass 实现。不要按名字推断两套方言语义完全相同。

## 3. 追踪一次 lowering

写出真实转换链路：

```text
高层 Conv（模型级语义） → lowering pass → 更低层计算/目标表示
```

每个箭头回答：哪些信息展开或固定了？哪些仍在 IR 中？shape 是否变化？失败时合法性前提是什么？不要假定一次 lowering 同时完成量化、内存规划和代码生成。

源码阅读时分清三个问题：实现在哪里、如何注册、当前 pipeline 是否真的调用。找到一个 pass 类不代表运行时一定执行它。

## 4. TOSA 与项目方言比较

分别填写下列信息并附源码位置：Op 定义/验证、shape/layout 表达、量化表示、lowering 目标。若只选了其中一种，说明选择原因，并将另一种留作对照阅读，而不是臆测其行为。

## 动手练习

- 给选定 Op 制作一张说明卡。
- 从 pass 注册名追到实现和替换/创建 Op 的代码，记录输入与输出方言。
- 与 `linalg.generic` 或本项目 `mynpu.conv` 对照：高层形式保留了什么信息？lowering 后什么更显式？

## 常见误会

- 高层 Dialect 不只是换一套名字，通常还保留有用语义和约束。
- TOSA 和 TPU-MLIR Top 是不同方言；是否经过 TOSA 要看真实 pipeline。
- Conv 被 lowering 不代表后端代码生成和部署已完成。

## 自测

1. 所选 Op 保留哪些模型语义？
2. verifier 检查哪些条件？
3. 哪个 pass 执行转换，输出是什么？

## 参考资料

- [TOSA Dialect](https://mlir.llvm.org/docs/Dialects/TOSA/)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
- TPU-MLIR：以当前 checkout 的文档、Op 定义、测试和 pipeline 为准。
