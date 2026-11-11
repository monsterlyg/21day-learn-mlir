# Day 20：TPU-MLIR 源码地图——追踪一个算子

## 今天的目标

围绕一个 Conv 或 Add，找到它从模型导入到目标 lowering 的关键入口。产出一张带文件、类/函数、pass 名和证据的源码地图，而不是试图读完整个编译器。

TPU-MLIR 目录和 pass 会随版本变化。记录当前 checkout 的 commit；若本仓库不是 TPU-MLIR，则在可访问的 checkout 上完成并注明范围。

## 1. 找五个锚点

对选定 Op 搜索并记录：

1. 定义：ODS/TableGen、C++ 类、属性和 verifier；
2. 导入：ONNX 前端如何构造 Op、设置属性和权重；
3. 优化：融合、canonicalization、shape/layout 变换；
4. 量化：校准、参数传播和类型/Op 改写入口；
5. lowering：高层到中间表示、再到目标相关表示的 pass。

可先用 `rg` 搜 Op 名、类名、注册名和 pass 关键字；随后确认 pipeline 的调用顺序。

## 2. 跟踪信息变化

按实际项目阶段画图，不预设固定 pass 名称：

```text
模型文件 → 导入 IR → 优化/量化 → 中间 IR → 目标相关 IR → 部署产物
```

每一步追踪 Conv 的 layout、padding/stride/dilation、shape、scale/zero point，以及何时选择硬件实现。每项附源码位置，并尽量用实际 IR dump 佐证。

## 3. 分清实现、注册、执行

对关键 pass 分别回答：

```text
实现在哪里？ → 如何注册？ → 当前命令/配置何时调用？
```

仅找到 pass 类说明有实现，不证明当前工具注册或执行了它。要从命令入口和 pipeline 构造追踪调用。

## 动手练习

- 为 Conv/Add 写 Op 卡片：语义、I/O、属性、shape 规则、源码位置。
- 运行最小模型，按本版本支持的选项保存至少三个阶段 IR。
- 选择一个 IR 属性变化，找到负责改写它的 pass 及 pipeline 调用点。找不到就列为待查，不作推断。

## 常见误会

- 文件名带 Lower 不代表它会在当前运行执行。
- 其他版本的教程路径可能已经变动。
- 量化、布局、内存、代码生成可能分散在多个阶段。

## 自测

1. Op 在哪里定义和导入？
2. 哪些 pass 处理优化、量化和 lowering？
3. 如何确认 pass 实际进入当前 pipeline？

## 参考资料

- TPU-MLIR：以当前 checkout 的 README、测试、源码和工具帮助为准。
- [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
