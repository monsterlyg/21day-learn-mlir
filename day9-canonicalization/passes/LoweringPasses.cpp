#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "IR/MyNPU.h"
#include "IR/MyNPUOps.h"
#include "PassesPlugin.h"

using namespace mlir;

namespace {

class MyNPUAddfLowering : public OpConversionPattern<mynpu::AddfOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        mynpu::AddfOp op,
        OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<arith::AddFOp>(
            op,
            adaptor.getOperands()[0],
            adaptor.getOperands()[1]);

        return success();
    }
};

struct LowerMyNPUToArithPass : public PassWrapper<
                                    LowerMyNPUToArithPass,
                                    OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMyNPUToArithPass)

    StringRef getArgument() const final { return "lower-mynpu-to-arith"; }
    StringRef getDescription() const final{
        return "Lower MyNPU dialect to arithmetic dialect";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect>();
    }

    void runOnOperation() override {
        MLIRContext &context = getContext();

        ConversionTarget target(context);

        // mynpu 最终不允许存在
        target.addIllegalDialect<mynpu::MyNPUDialect>();

        // arith 可以存在
        target.addLegalDialect<arith::ArithDialect>();

        // func/module 允许存在
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalOp<ModuleOp>();

        RewritePatternSet patterns(&context);

        patterns.add<MyNPUAddfLowering>(&context);

        if (failed(applyFullConversion(
                getOperation(),
                target,
                std::move(patterns)))){
            signalPassFailure();
        }
    }
};

} // namespace

void registerMyNPULoweringPasses(){
    PassRegistration<LowerMyNPUToArithPass>();
}
