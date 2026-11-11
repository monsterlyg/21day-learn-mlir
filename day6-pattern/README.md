# day5 — Rewrite Pattern：从子图匹配到算子融合

上一 day（day4）我们学会了写一个**自定义 Pass** 并编译成插件、用 `mlir-opt` 加载运行。
这一 day 我们要学 MLIR 里最常用的变换手段——**Rewrite Pattern**，并用它做**算子融合**：
把「卷积 → 残差加 → ReLU」「卷积 → BN → ReLU」这两段子图，融合成一个**自定义的具名算子**。

## 1. 为什么基础方言选 Linalg，而不是 ONNX

你更熟悉 ONNX 算子，但**原生 MLIR 不包含 ONNX 方言**（它由 `mlir-onnx` 项目提供，不在
`llvm-project` 主仓库里）。所以要用一个 MLIR 自带的、适合做算子融合的方言作为学习基础。

**Linalg** 是 MLIR 做结构化算子融合的标准方言，理由：

- `linalg.conv_2d_nhwc_hwcf` / `linalg.add` / `linalg.mul` 都是**具名结构化算子**，
  语义清晰，正好对应 ONNX 的 `Conv` / `Add` / `Mul`；
- `linalg.generic` 是通用算子，可以用来表达 ReLU 这类 element-wise 操作；
- Linalg 本身就是 MLIR 做 `tiling` / `fusion` 的主战场，和你要学的「子图匹配→融合」
  直接对口。

> 注意：`linalg.conv_2d_nhwc_hwcf` 的权重布局是 **HWCF = `(KH, KW, C, F)`**，即
> `tensor<3x3x64x64xf32>`，**不是** `64x64x3x3`。`nhwc` 指输入/输出的通道在最后一维。

## 2. 核心概念：Rewrite Pattern

一个 Rewrite Pattern 回答两个问题：**匹配什么子图**（`match`）、**重写成什么**（`rewrite`）。
MLIR 把它抽象成一个类：

```cpp
struct MyPattern : public OpRewritePattern<OpTy> {
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    // 1. 判断 op 及其上下文是否命中目标子图，不命中 return failure();
    // 2. 命中则 rewriter.create / rewriter.replaceOp 完成重写，return success();
  }
};
```

- **`OpRewritePattern<OpTy>`**：把「锚点 op」固定成 `OpTy`。贪心驱动会拿每个 `OpTy` 实例
  来问这个 pattern 是否匹配。`op` 参数就是当前被考察的那个 `OpTy`。
- **`matchAndRewrite`**：匹配 + 重写二合一。返回 `success()` 表示这次被改写了，
  `failure()` 表示不匹配、跳过。
- **`PatternRewriter`**：写操作的唯一入口（`create`、`replaceOp`、`eraseOp`、`setInsertionPoint`…）。
  所有对 IR 的修改都必须通过它，它负责记录改动、维护 use-def 关系。
- **`applyPatternsGreedily`**：贪心驱动，反复扫描整个函数，把 patterns 集合应用直到
  **不动点**（没有 pattern 再命中）。它还自带 **dead code elimination（DCE）**：我们只
  `replaceOp` 掉最后的 relu，中间变孤儿（没使用者）的 conv/add 会被自动清掉。

> ⚠️ 版本坑：`applyPatternsAndFoldGreedily` 在 MLIR 23 里**已删除**，改用
> `applyPatternsGreedily`（头文件 `mlir/Transforms/GreedyPatternRewriteDriver.h`，
> 链接库 `MLIRTransforms`）。

## 3. 三个 Pattern 详解

### Pattern 1：`FoldAddZeroPattern`（`--fold-add-zero`）—— 讲清骨架

最简例子：折叠 `x + 0.0` → `x`。

```cpp
struct FoldAddZeroPattern : public OpRewritePattern<arith::AddFOp> {
  LogicalResult matchAndRewrite(arith::AddFOp op, PatternRewriter &rewriter) const override {
    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {   // 右操作数是 +0.0？
      rewriter.replaceOp(op, op.getLhs());               // x + 0.0 -> x
      return success();
    }
    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {   // 0.0 + x -> x
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};
```

- `m_AnyZeroFloat()` 是 matcher（`mlir/IR/Matchers.h`），能匹配常量 `+0.0` / `-0.0`。
- `rewriter.replaceOp(op, newValue)`：把所有用 `op` 结果的地方换成 `newValue`，并擦除 `op`。

### Pattern 2：`ConvAddReluFusionPattern`（`--fuse-conv-add-relu`）

匹配子图：

```
%conv = linalg.conv_2d_nhwc_hwcf ins(%input, %weight) ...
%add  = linalg.add ins(%conv, %residual) ...     // 残差连接
%relu = linalg.generic { arith.maximumf(%, 0.0) } // ReLU
```

思路（**正向匹配**，锚在 conv 上，顺着 `getUsers()` 往后找）：

1. conv 的输出**只有一个使用者**（`hasOneUse()`），且它是 `linalg::AddOp`；
2. add 的输出只有一个使用者，且它是 `linalg::GenericOp`，且 body 是 ReLU
   （`isRelu()` 校验：1 输入 1 输出、yield 的是 `arith.maximumf`）；
3. 取出三个输入：`input = conv 的第 0 个输入`、`weight = conv 的第 1 个输入`、
   `residual = add 的另一个输入`（不是 conv 输出的那个）；
4. 建 `fusion.conv_add_relu`，`replaceOp(relu, fused)`。conv/add 变孤儿，被贪心 DCE 清掉。

`isRelu()` 判断一个 `linalg.generic` 是不是 ReLU：

```cpp
static bool isRelu(linalg::GenericOp op) {
  if (op.getDpsInputs().size() != 1 || op.getDpsInits().size() != 1)
    return false;
  Block &block = op.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  return llvm::isa<arith::MaximumFOp>(yield.getOperand(0).getDefiningOp());
}
```

> 版本坑：Linalg 老接口 `getInputs()` / `getOutputs()` 已删除，改用
> `getDpsInputs()`（返回 `SmallVector<Value>`）和 `getDpsInits()`（返回 `OperandRange`）。

### Pattern 3：`ConvBnReluFusionPattern`（`--fuse-conv-bn-relu`）

同构，只是链更长：`conv → mul(scale) → add(shift) → relu`（把 BN 简化为「乘 scale + 加 shift」），
融合成 `fusion.conv_bn_relu(input, weight, scale, shift)`。匹配逻辑完全对应 Pattern 2，
只是多了「`linalg::MulOp`」这一段。

## 4. 自定义融合算子：ODS / TableGen

因为我们选了「融合成一个**自定义具名 op**」，所以 day5 必须引入 **ODS（Operation Definition
Specification）/ TableGen**——MLIR 用声明式 DSL 自动生成 op 的声明、构建器、parse/print/verify。

`FusionOps.td` 里先声明方言，再声明两个 op：

```tablegen
def Fusion_Dialect : Dialect {
  let name = "fusion";                    // 文本里写成 fusion.xxx
  let cppNamespace = "::mlir::fusion";    // C++ 里 mlir::fusion 命名空间
}

def Fusion_ConvAddReluOp : Fusion_Op<"conv_add_relu", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$weight, AnyRankedTensor:$residual);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{ $input `,` $weight `,` $residual attr-dict `:`
                          type($input) `,` type($weight) `,` type($residual) `->` type($result) }];
}
```

- `[Pure]`：无副作用、可被 CSE/DCE 安全处理（`SideEffectInterfaces.td`）。
- `assemblyFormat`：自定义的打印/解析格式，让 op 打印成
  `fusion.conv_add_relu %in, %w, %res : tensor<...>, tensor<...>, tensor<...> -> tensor<...>`。

CMake 里 `add_mlir_dialect(FusionOps fusion)` 会用 `mlir-tblgen` 从 `.td` 生成一堆 `.inc`
（`FusionOps.h.inc` / `FusionOps.cpp.inc` / `FusionOpsDialect.h.inc` / …），我们手写的
`FusionDialect.h/.cpp`、`FusionOps.h/.cpp` 只是把它 `#include` 进来。这些骨架照抄官方
`mlir/examples/standalone/` 即可。

## 5. 两个插件入口

Pass 会 `create` 出 `fusion.*` op，所以运行时 **fusion 方言必须已注册进 context**。两种方式：

```cpp
// 入口 1：方言插件（注册方言 + pass），用 --load-dialect-plugin 加载
extern "C" mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Fusion", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<fusion::FusionDialect>();
            registerFusionPasses();
          }};
}
// 入口 2：纯 pass 插件，用 --load-pass-plugin 加载
extern "C" mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "FusionPasses", LLVM_VERSION_STRING,
          []() { registerFusionPasses(); }};
}
```

- 会创建 `fusion.*` 的两个 pass 用 `--load-dialect-plugin`（顺便把方言注册进 context）。
- `fold-add-zero` 不碰 fusion 方言，用 `--load-pass-plugin` 即可。
- `getDependentDialects()` 声明 pass 依赖哪些方言，让 `--load-pass-plugin` 时方言能被按需加载。

## 6. 编译

```bash
cd /data8/liyinggang/Projects/CXX/mlir-learn/day5
mkdir -p build && cd build
/data8/liyinggang/Assets/cmake-3.29.8-linux-x86_64/bin/cmake ..   # 系统 cmake 3.16 太旧，必须用 3.29
make -j4
# 产出 build/libFusionPlugin.so
```

CMake 要点（完整见 `CMakeLists.txt`）：

- `include(TableGen)`、`include(AddLLVM)`、`include(AddMLIR)`，并把 `${MLIR_CMAKE_DIR}`、
  `${LLVM_CMAKE_DIR}` 加进 `CMAKE_MODULE_PATH`。
- **`include_directories(${MLIR_INCLUDE_DIRS} ...)` 必须放在 `add_mlir_dialect` 之前**——
  `mlir-tblgen` 靠读 `INCLUDE_DIRECTORIES` 目录属性来找 `mlir/IR/OpBase.td`，顺序反了会报
  `could not find include file 'mlir/IR/OpBase.td'`。
- 生成的 `.cpp.inc` 里用到 `OpBuilder`/`ImplicitLocOpBuilder` 的完整定义，`FusionOps.cpp`
  需要额外 `#include "mlir/IR/Builders.h"`。

## 7. 运行与验证

```bash
M=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
P=./build/libFusionPlugin.so

# [1] 折叠 x + 0.0 -> x
$M --load-pass-plugin $P -p "builtin.module(fold-add-zero)" test/add_zero.mlir

# [2] conv + add + relu -> fusion.conv_add_relu（先看 before 再看 after）
$M test/resnet_block.mlir
$M --load-dialect-plugin $P -p "builtin.module(fuse-conv-add-relu)" test/resnet_block.mlir

# [3] conv + bn + relu -> fusion.conv_bn_relu
$M --load-dialect-plugin $P -p "builtin.module(fuse-conv-bn-relu)" test/resnet_block_bn.mlir
```

验证三件事（也写进了 `test/check.sh`，直接 `bash test/check.sh` 一键跑）：

1. **before/after**：不带 pass 的输出 vs 带 pass 的输出，确认 `linalg.conv_2d` + `linalg.add`
   + ReLU `linalg.generic` 三个 op → 单个 `fusion.conv_add_relu`；
2. **维度正确**：input `1x56x56x64`、weight `3x3x64x64`、residual/输出 `1x54x54x64`
   （无 padding、stride 1、kernel 3×3 → 输出尺寸 56−3+1=54）；
3. 三个输入 `.mlir` 能被 `mlir-opt` 直接 round-trip（不报 parse 错误）。

### ⚠️ 一个重要坑：插件 pass 不能用 `--pass-name`，必须用 `-p`

你可能会写成 `mlir-opt --load-pass-plugin $P --fold-add-zero ...`，会得到：

```
mlir-opt: Unknown command line argument '--fold-add-zero'.
```

原因：`mlir-opt` 在**解析命令行之前**就把「已注册 pass 的名字」固化成 cl 字面量选项表；
而 `--load-pass-plugin` 是在**解析过程中**才 `dlopen` 插件、注册 pass 的——为时已晚。
所以插件 pass 必须走文本流水线 `-p`，且因为我们的 pass 作用在 `builtin.module` 上，要包锚点：

```bash
-p "builtin.module(fold-add-zero)"
```

`--help` / `--list-passes` 能看到插件 pass（它们在解析到 `--help` 时才枚举注册表），但这不代表
能当 `--pass-name` 用。这是插件机制本身的限制，不是我们代码的问题。

## 8. 扩展：真实的 BN（逐通道 + broadcast）

本例把 BN 简化成了「同 shape 的整张 `scale`/`shift`」，便于理解匹配链。真实 BN 是**逐通道**
的：`scale`/`shift` 是 `tensor<64xf32>`（对最后一维 C 做 broadcast）。要在 `linalg.mul` 的
输入里出现 `tensor<64xf32>` 与 `tensor<1x54x54x64xf32>` 做 broadcast，需要改用带
`indexing_maps` 的 `linalg.generic` 来表达 scale/shift（而不是 `linalg.mul`/`linalg.add`
的简单具名形式），pattern 里对「BN」的识别也要相应升级：识别 generic 的 body 是
`%y = arith.mulf %x, %scale_channel` / `%z = arith.addf %y, %shift_channel`。这是下一步
（day6）可以做的进阶练习。

## 9. 文件结构

```
day5/
├── CMakeLists.txt          # TableGen + AddMLIR + 生成 .inc + 链接 linalg/tensor/transforms
├── FusionOps.td            # ODS：fusion 方言 + 两个具名融合 op
├── FusionDialect.h/.cpp    # 方言类（include 生成的 *Dialect.h.inc / *.cpp.inc）
├── FusionOps.h/.cpp        # op 类（include 生成的 *Ops.h.inc / *.cpp.inc）
├── FusionPasses.h/.cpp     # 3 个 OpRewritePattern + 3 个 Pass + registerFusionPasses()
├── FusionPassesPlugin.cpp  # mlirGetDialectPluginInfo + mlirGetPassPluginInfo
└── test/
    ├── add_zero.mlir
    ├── resnet_block.mlir
    ├── resnet_block_bn.mlir
    └── check.sh
```
