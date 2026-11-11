//===- FusionPasses.cpp - rewrite patterns + passes --------------*- C++ -*-===//
//
// day5 核心：三个 Rewrite Pattern，从「折叠」到「子图融合」。
//
//   Pattern 1 (FoldAddZeroPattern)：最简 OpRewritePattern，讲清骨架。
//   Pattern 2 (ConvAddReluFusionPattern)：正向匹配 conv -> add -> relu，
//             融合成单个 fusion.conv_add_relu。
//   Pattern 3 (ConvBnReluFusionPattern)：正向匹配 conv -> mul -> add -> relu
//             （BN + relu），融合成单个 fusion.conv_bn_relu。
//
// 匹配方向：正向（锚在 conv 上，顺着 getUsers() 往后找），符合
// 「先找到 conv，再往后识别 add/relu」的直觉。
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "FusionDialect.h"
#include "FusionOps.h"
#include "FusionPasses.h"

using namespace mlir;

// ---------------------------------------------------------------------------
// helper：判断一个 linalg.generic 是不是 relu。
//   relu = 1 个输入 + 1 个输出，region 里 yield 的是 arith.maximumf。
// ---------------------------------------------------------------------------
static bool isRelu(linalg::GenericOp op) {
  if (op.getDpsInputs().size() != 1 || op.getDpsInits().size() != 1)
    return false;
  Block &block = op.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  return llvm::isa<arith::MaximumFOp>(yield.getOperand(0).getDefiningOp());
}

namespace {

// ---------------------------------------------------------------------------
// Pattern 1：折叠 arith.addf(x, 0.0) -> x
// 讲清 RewritePattern 骨架：matchAndRewrite + rewriter.replaceOp。
// ---------------------------------------------------------------------------
struct FoldAddZeroPattern : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    // 右操作数是 +0.0：x + 0.0 -> x
    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      // replaceOp 用一个或多个新值替换旧操作的结果，然后删除旧操作
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // 左操作数是 +0.0：0.0 + x -> x
    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

// ---------------------------------------------------------------------------
// Pattern 2：conv -> add(残差) -> relu  ==>  fusion.conv_add_relu
// ---------------------------------------------------------------------------
struct ConvAddReluFusionPattern
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter &rewriter) const override {
    // 1) conv 的输出必须只有一个使用者：那个 add
    Value convOut = conv.getResult(0);
    if (!convOut.hasOneUse())
      return failure();
    auto add = llvm::dyn_cast<linalg::AddOp>(*convOut.getUsers().begin());
    if (!add)
      return failure();

    // 2) add 的输出必须只有一个使用者：那个 relu
    Value addOut = add.getResult(0);
    if (!addOut.hasOneUse())
      return failure();
    auto relu = llvm::dyn_cast<linalg::GenericOp>(*addOut.getUsers().begin());
    if (!relu || !isRelu(relu))
      return failure();

    // 3) 取出三个输入：conv 的 input/weight，以及 add 的另一个输入(残差)
    SmallVector<Value> convIns = conv.getDpsInputs();
    SmallVector<Value> addIns = add.getDpsInputs();
    Value input = convIns[0];
    Value weight = convIns[1];
    Value residual = (addIns[0] == convOut) ? addIns[1] : addIns[0];

    // 4) 用 fused op 替换 relu；conv/add 变成死代码，由贪心驱动的 DCE 清除
    rewriter.setInsertionPoint(relu);
    auto fused = fusion::ConvAddReluOp::create(
        rewriter, conv.getLoc(), relu.getResultTypes(),
        ValueRange{input, weight, residual});
    rewriter.replaceOp(relu, fused.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pattern 3：conv -> bn -> relu  ==>  fusion.conv_bn_relu
// ---------------------------------------------------------------------------
struct ConvBnReluFusionPattern : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp conv,
                                PatternRewriter& rewriter) const override {
    // 1) conv 的输出必须只有一个使用者：那个 mul
    Value conv_out = conv.getResult(0);
    if (!conv_out.hasOneUse()) return failure();
    
    // 用 mul 来表示 BN 的 scale
    auto mul = llvm::dyn_cast<linalg::MulOp>(*conv_out.getUsers().begin());
    if (!mul) return failure();

    // 2) mul 的输出必须只有一个使用者：那个 relu
    Value mul_out = mul.getResult(0);
    if (!mul_out.hasOneUse()) return failure();
    auto relu = llvm::dyn_cast<linalg::GenericOp>(*mul_out.getUsers().begin());
    if (!relu || !isRelu(relu)) return failure();

    // 3) 取出三个输入：conv 的 input/weight，以及 mul 的另一个输入(scale)
    SmallVector<Value> convIns = conv.getDpsInputs();
    SmallVector<Value> mulIns = mul.getDpsInputs();
    Value input = convIns[0];
    Value weight = convIns[1];
    Value scale = (mulIns[0] == conv_out) ? mulIns[1] : mulIns[0];

    // 4) 用 fused op 替换 relu；conv/add 变成死代码，由贪心驱动的 DCE 清除
    rewriter.setInsertionPoint(relu);
    auto fused = fusion::ConvBnReluOp::create(
        rewriter, conv.getLoc(), relu.getResultTypes(),
        ValueRange{input, weight, scale});
    rewriter.replaceOp(relu, fused.getResult());

    return success();
  }
};


// ---------------------------------------------------------------------------
// 三个 Pass（沿用 day4 的 PassWrapper 风格）
// ---------------------------------------------------------------------------
struct FoldAddZeroPass
    : public PassWrapper<FoldAddZeroPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddZeroPass)

  StringRef getArgument() const final { return "fold-add-zero"; }
  StringRef getDescription() const final {
    return "Fold arith.addf(x, 0.0) into x";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAddZeroPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct FuseConvAddReluPass
    : public PassWrapper<FuseConvAddReluPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseConvAddReluPass)

  StringRef getArgument() const final { return "fuse-conv-add-relu"; }
  StringRef getDescription() const final {
    return "Fuse conv + residual add + relu into fusion.conv_add_relu";
  }

  // 声明本 pass 会创建 fusion.* op，需要 fusion 方言；同时用到 linalg/arith。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<fusion::FusionDialect, linalg::LinalgDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvAddReluFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct FuseConvBnReluPass
    : public PassWrapper<FuseConvBnReluPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseConvBnReluPass)

  StringRef getArgument() const final { return "fuse-conv-bn-relu"; }
  StringRef getDescription() const final {
    return "Fuse conv + batchnorm mul + relu into fusion.conv_bn_relu";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<fusion::FusionDialect, linalg::LinalgDialect,
                    arith::ArithDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvBnReluFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};


} // namespace

void registerFusionPasses() {
  PassRegistration<FoldAddZeroPass>();
  PassRegistration<FuseConvAddReluPass>();
  PassRegistration<FuseConvBnReluPass>();
}
