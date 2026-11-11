# Day 17：模型导入——把 ONNX 变成可检查的 IR

## 今天的目标

把 importer 当作现成前端使用：导入一个小 ONNX 模型，逐节点检查产生的 IR，并理解 shape、属性和权重如何进入编译器。今天不实现 ONNX importer。

## 1. 准备最小模型

选少量节点的模型，例如 `Conv → Relu` 或 `MatMul → Add`。记录模型来源、opset（若已知）、输入名称/shape/dtype、权重配置。优先使用仓库自带示例，避免同时引入复杂预处理。

若使用 TPU-MLIR，先查当前版本文档及 `model_transform.py` 的帮助，确认参数；工具选项会随版本变化。示意命令如下，必须按本机脚本调整：

```sh
python3 model_transform.py --model_name tiny --model_def tiny.onnx \
  --input_shapes '[1,3,8,8]' --mlir tiny.mlir
```

保存版本、精确命令、产物路径，以及 warning、unsupported op 和 shape 假设。如果环境没有 importer，使用已生成 IR 完成分析，并注明没有运行导入步骤。

## 2. 建立节点映射表

| ONNX 节点 | 输入/输出 | 导入后的 MLIR Op | 附加属性 |
|---|---|---|---|
| Conv | 按模型填写 | 按产物填写 | stride、pad、layout 等 |
| Relu | 按模型填写 | 按产物填写 | 类型、shape |

逐项核对：节点是否保留或拆分？权重是 SSA 常量、属性还是外部引用？ONNX 属性映射到哪里？shape 来自模型、命令行还是推导？是否插入 transpose、reshape、cast 等辅助 Op？

导入后的节点数不一定等于 ONNX 节点数；要按数据流和语义核对，而不是只数 Op。

## 3. 独立验证 shape

选择 Conv 的一个输出维度，用输入 shape、kernel、stride 和 padding 手算，再对照 IR 类型。若不同，先查 layout、自动 padding、动态维度和导入参数，不要直接假定一边错误。

## 动手练习

### A. 导入并留档

保存模型、命令、工具版本、导入后 IR 和日志。

### B. 画 SSA 数据流

标出运行时输入、常量权重、Conv、激活和输出。

### C. 改一个输入 shape

对另一组合法静态 shape 重复导入，比较类型、属性和图结构。解释变化来自 shape 传播还是模型结构。

## 常见误会

- 导入 IR 不是 ONNX 的逐字翻译，可能有辅助 Op 或规范化。
- shape 不一定都由模型文件确定，动态维度和 CLI 参数也可能参与。
- 导入成功不等于数值语义已完全验证；还要核对 layout、权重、dtype 和 unsupported 算子。

## 自测

1. 一个 ONNX 节点对应哪些 IR Op？
2. 权重和运行时输入如何区分？
3. shape 信息从哪些来源进入 IR？

## 参考资料

- ONNX：查模型使用的 opset 文档及 importer 文档。
- TPU-MLIR：查当前 checkout 的导入指南和脚本帮助。
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
