//===- MyNPUOps.cpp - MyNPU dialect ops ---------------------*- C++ -*-===//
//
// Op 方法实现（仅包含 TableGen 生成的代码，接口实现在 MyNPU.cpp）。
//
//===----------------------------------------------------------------------===//

#include "MyNPUOps.h"
#include "mlir/IR/PatternMatch.h"

mlir::LogicalResult mynpu::ConvOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(operands[1].getType());
    if (!inputType || !weightType)
        return mlir::failure();

    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();
    if (inputShape.size() != 4 || weightShape.size() != 4)
        return mlir::failure();

    if (!properties)
        return mlir::failure();
    auto convProperties = properties.as<ConvOp::Properties *>();
    if (!convProperties)
        return mlir::failure();

    auto strideAttr = convProperties->stride;
    auto paddingAttr = convProperties->padding;
    if (!strideAttr || !paddingAttr || strideAttr.size() != 2 ||
        paddingAttr.size() != 2)
        return mlir::failure();

    int64_t sh = mlir::cast<mlir::IntegerAttr>(strideAttr[0]).getInt();
    int64_t sw = mlir::cast<mlir::IntegerAttr>(strideAttr[1]).getInt();
    int64_t ph = mlir::cast<mlir::IntegerAttr>(paddingAttr[0]).getInt();
    int64_t pw = mlir::cast<mlir::IntegerAttr>(paddingAttr[1]).getInt();

    int64_t oh = (inputShape[2] + 2 * ph - weightShape[2]) / sh + 1;
    int64_t ow = (inputShape[3] + 2 * pw - weightShape[3]) / sw + 1;
    auto outputType = mlir::RankedTensorType::get(
        {inputShape[0], weightShape[0], oh, ow}, inputType.getElementType());
    inferredReturnTypes.push_back(outputType);
    return mlir::success();
    }

int64_t mynpu::ConvOp::getComputeCost() {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(getInput().getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(getWeight().getType());
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    if (!inputType || !weightType || !outputType || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape() || !outputType.hasStaticShape())
        return 0;

    // MACs * 2 (one multiply and one add), using NCHW input and OIHW weights.
    return outputType.getNumElements() * inputType.getDimSize(1) *
            weightType.getDimSize(2) * weightType.getDimSize(3) * 2;
}

int64_t mynpu::AddfOp::getComputeCost() {
    auto outputType = mlir::dyn_cast<mlir::ShapedType>(getResult().getType());
    if (!outputType || !outputType.hasStaticShape())
        return 0;
    return outputType.getNumElements();
}

int64_t mynpu::ReluOp::getComputeCost() {
    auto outputType = mlir::dyn_cast<mlir::ShapedType>(getOutput().getType());
    if (!outputType || !outputType.hasStaticShape())
        return 0;

    // One max/select per output element.
    return outputType.getNumElements();
}

struct RemoveNestedReluPattern : mlir::OpRewritePattern<mynpu::ReluOp> {
    using OpRewritePattern<mynpu::ReluOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mynpu::ReluOp reluOp,
                                        mlir::PatternRewriter &rewriter) const override {
        auto pre_op = reluOp.getInput().getDefiningOp<mynpu::ReluOp>();
        if (!pre_op)
            return mlir::failure();
        if (pre_op.getAxis() != reluOp.getAxis())
            return mlir::failure();
        if (pre_op.getResult().getType() != reluOp.getResult().getType())
            return mlir::failure();
        rewriter.replaceOp(reluOp, pre_op.getResult());
        return mlir::success();
    }
};

void mynpu::ReluOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet & patterns, mlir::MLIRContext* context) {
    patterns.add<RemoveNestedReluPattern>(context);
}

#define GET_OP_CLASSES
#include "MyNPUOps.cpp.inc"
