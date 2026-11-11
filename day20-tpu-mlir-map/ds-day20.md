# Day 20：TPU-MLIR 源码地图——一个 Op 的五个入口

## 今天要解决什么问题？

拿到一个陌生的编译器仓库，最常见的问题不是「某个 API 怎么用」，而是：

> 我想看 `Conv` 从导入到目标代码的全过程，应该从哪个文件开始？改一个行为，要动哪几层？

今天不追某个具体实现细节，而是建立一张**可复用的源码地图**。目标是：给你一个算子名，你能在 10 分钟内找到它的定义、导入、优化、量化和 lowering 入口。

本机没有 TPU-MLIR 的 checkout，所以今天分两部分：

1. 用本机 `llvm-project` 里的 TOSA 做一次完整演练（命令可跑）。
2. 把同一套方法映射到 TPU-MLIR 的 Top/Tpu 结构（路径以你 clone 的版本为准）。

## 环境

```bash
MLIR_ROOT=/data8/liyinggang/Projects/CXX/llvm-project
MLIR_OPT=$MLIR_ROOT/lyg-install/bin/mlir-opt
```

本机 TOSA 源码位置（已验证）：

```text
mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td
mlir/lib/Dialect/Tosa/IR/TosaOps.cpp
mlir/lib/Dialect/Tosa/IR/TosaCanonicalizations.cpp
mlir/lib/Dialect/Tosa/Transforms/
mlir/lib/Conversion/TosaToLinalg/
mlir/include/mlir/Conversion/Passes.td
mlir/test/Dialect/Tosa/
mlir/test/Conversion/TosaToLinalg/
```

## 第一站：五个入口

对一个高层算子，沿着信息流找五个地方：

```text
1. Op 定义      这个算子有哪些操作数、结果、属性、接口？
2. 导入/构造    模型里的节点怎样变成这个 Op？
3. 优化         这个 Op 有哪些 canonicalize / fold / 专用 Pass？
4. 量化         量化信息挂在哪里？是否有专用转换？
5. Lowering     它被转换到哪个更低层方言？转换模式在哪？
```

这五问对应 MLIR 里已经学过的机制：

| 入口 | 对应机制 | 学过的地方 |
|---|---|---|
| Op 定义 | ODS / TableGen | Day 6 |
| 导入 | 前端构造 op / builder | Day 17 |
| 优化 | canonicalizer / fold / Pass | Day 9、Day 18 |
| 量化 | quant 类型 / 量化 Pass | Day 19 |
| Lowering | Dialect Conversion / Pattern | Day 7 |

## 第二站：用 TOSA 练一遍

以 `tosa.conv2d` 为例。

### 1. Op 定义

```bash
rg -n "def Tosa_Conv2DOp" $MLIR_ROOT/mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td
```

打开后重点看：

- `arguments`：input、weight、bias、input_zp、weight_zp。
- `results`：output。
- `traits` / `interfaces`：是否实现 shape inference。
- `hasVerifier` / `hasCanonicalizer`：有没有额外校验和规范化。

### 2. 优化入口

```bash
rg -n "Conv2D|conv2d" $MLIR_ROOT/mlir/lib/Dialect/Tosa/IR/TosaCanonicalizations.cpp
rg -n "conv2d" $MLIR_ROOT/mlir/lib/Dialect/Tosa/Transforms/
```

不是每个 op 都有专用 canonicalization；有些只依赖泛化的 fold/DCE。找到就记录，找不到也是结论。

### 3. Lowering 入口

```bash
rg -n "Conv2DOp" $MLIR_ROOT/mlir/lib/Conversion/TosaToLinalg/
```

你会看到 named lowering 和 generic lowering 两条路径：

```text
TosaToLinalgNamed.cpp  → linalg.conv_2d_nhwc_fhwc
TosaToLinalg.cpp       → linalg.generic + 基础张量操作
```

### 4. Pass 注册

```bash
rg -n "TosaToLinalgNamed|TosaToLinalg\b" $MLIR_ROOT/mlir/include/mlir/Conversion/Passes.td
```

这里能看到 pass 的 anchor 类型，也是 Day 16 那个 `func.func(...)` 要求的来源。

### 5. 测试

```bash
ls $MLIR_ROOT/mlir/test/Conversion/TosaToLinalg/
ls $MLIR_ROOT/mlir/test/Dialect/Tosa/
```

测试是最好的「用法说明书」：它同时给出输入 IR、运行命令和期望输出。

## 第三站：TPU-MLIR 的对应布局

下面是 TPU-MLIR 常见的目录组织。**不同版本可能有调整**，用 `rg` 核对，不要盲信路径。

```text
include/tpu_mlir/Dialect/Top/IR/        # 高层算子定义（类似 TOSA）
lib/Dialect/Top/IR/                    # 高层算子的 verifier/canonicalizer
include/tpu_mlir/Dialect/Tpu/IR/       # 目标相关算子定义
lib/Dialect/Tpu/IR/
lib/Dialect/Top/Transforms/            # 高层优化 Pass
lib/Conversion/TopToTpu/               # Top → Tpu lowering
tools/model_transform.py               # 模型导入入口
tools/run_mlir.py                      # pipeline 驱动
test/                                  # 端到端和转换测试
python/                                # Python 工具与绑定
```

定位命令：

```bash
# 1. Op 定义
rg -n "def Top_Conv|class ConvOp" include lib

# 2. 导入
rg -n "Conv" tools/model_transform.py python/ | head -30

# 3. 优化
ls lib/Dialect/Top/Transforms/
rg -n "Conv" lib/Dialect/Top/Transforms/

# 4. Top → Tpu lowering
rg -n "TopToTpu|class.*Conv" lib/Conversion/TopToTpu/ | head -40

# 5. 量化
rg -n "quant|Quant" lib/Conversion/TopToTpu/ lib/Dialect/Top/Transforms/ | head -40
```

记录版本，否则地图会过期：

```bash
git rev-parse HEAD
git log -1 --format='%h %ad %s'
```

## 第四站：填一张 Op 卡片

每读一个新算子，建议填同样的表。模板：

```markdown
## Op: <名字>

- 仓库 / commit：
- 高层方言与 op 名：
- 定义文件与行号：
- 操作数：
- 结果：
- 属性：
- traits / interfaces：
- verifier 位置：
- canonicalize / fold：
- 导入入口：
- 优化 Pass：
- 量化处理：
- lowering 目标与文件：
- 测试文件：
- 一句话语义：
```

以 `tosa.conv2d` 为例，填出来的关键字段是：

```text
定义：mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td
操作数：input(NHWC), weight(OHWI), bias, input_zp, weight_zp
属性：acc_type, pad, stride, dilation
lowering：linalg.conv_2d_nhwc_fhwc / linalg.generic
测试：mlir/test/Conversion/TosaToLinalg/
```

## 第五站：Top 和 Tpu 分别像什么？

用已经学过的概念类比：

| TPU-MLIR | 类比 | 职责 |
|---|---|---|
| `Top` 方言 | TOSA 高层算子 | 保住模型语义，做图级优化 |
| `Tpu` 方言 | 目标相关 IR | 表达芯片/后端相关能力和布局 |
| `TopToTpu` | Dialect Conversion | 把高层语义降到目标语义 |
| 量化 Pass | `quant` + 转换 | 插入量化信息、调整类型 |
| `run_mlir.py` | Pass pipeline 驱动 | 按阶段组织整个流程 |

看源码时，先问「这属于哪一层」，再问「它和上一层/下一层的契约是什么」。否则很容易陷进某个 pattern 的细节里。

## 今天的练习路线

### 练习 A：本地五问

对 `tosa.matmul` 重复上面的五问，填一张 Op 卡片。要求每个入口都给出文件路径和行号。

### 练习 B：找 anchor

打开 `mlir/include/mlir/Conversion/Passes.td`，找出 3 个 pass 的 anchor 类型（`Pass<...>` / `InterfacePass<...>` / `OperationPass<...>`），解释它们应该怎样写进 `-p`。

### 练习 C：读测试

从 `mlir/test/Conversion/TosaToLinalg/` 挑一个 conv 测试，回答：

1. 输入是什么？
2. 期望输出有哪些 op？
3. 它验证了哪条语义？

### 练习 D：TPU-MLIR 地图（有仓库时）

clone TPU-MLIR 后，用第三站的 `rg` 命令，填出 `Top.Conv` 的 Op 卡片。如果暂时没有仓库，先完成练习 A-C，并把 TPU-MLIR 部分留成待办。

## 常见坑

- **路径过期**：仓库重构很常见，先 `rg` 再打开文件。
- **生成代码和手写代码混淆**：`.td` 生成的头文件在 build 目录，不要改。
- **同名 op 多套方言**：`Conv` 可能在 Top、Tpu、ONNX、TOSA 里都存在。
- **只看定义不看测试**：测试能告诉你实际期望的 IR 形态。
- **忽略 commit**：源码地图必须绑定版本，否则下次找不到。

## 今天结束时，试着讲给别人听

1. 一个 Op 的五个入口分别是什么？
2. 为什么 Op 定义通常在 `.td`，而行为实现分散在多个 `.cpp`？
3. Top 和 Tpu 的分工像你学过的哪两个方言？
4. 怎样在 10 分钟内确认一个 pass 应该挂在哪个 anchor 上？
5. 为什么源码地图要记录 commit？

## 参考资料

- [TPU-MLIR](https://github.com/sophgo/tpu-mlir)
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)
- [MLIR Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- 本机源码：`mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td`、`mlir/lib/Conversion/TosaToLinalg/`
