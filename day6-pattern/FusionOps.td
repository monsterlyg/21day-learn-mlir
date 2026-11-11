//===- FusionOps.td - Fusion dialect ops -------------------*- tablegen -*-===//
//
// day6 学习用：自定义融合方言，定义两个具名融合算子，作为 Rewrite Pattern 的
// 融合目标，实现「三个 op -> 一个具名 op」。
//
//   fusion.conv_add_relu : relu(conv(x, w) + residual)
//   fusion.conv_bn_relu   : relu(conv(x, w) * scale + shift)
//
//===----------------------------------------------------------------------===//

#ifndef FUSION_OPS
#define FUSION_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Fusion dialect
//===----------------------------------------------------------------------===//

def Fusion_Dialect : Dialect {
  let name = "fusion";
  let summary = "A tiny dialect with fused conv+add/bn+relu ops, for learning rewrite patterns.";
  let cppNamespace = "::mlir::fusion";
}

class Fusion_Op<string mnemonic, list<Trait> traits = []>
    : Op<Fusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Fused ops
//===----------------------------------------------------------------------===//

def Fusion_ConvAddReluOp : Fusion_Op<"conv_add_relu", [Pure]> {
  let summary = "fused conv + residual add + relu";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       AnyRankedTensor:$residual);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $weight `,` $residual attr-dict
    `:` type($input) `,` type($weight) `,` type($residual) `->` type($result)
  }];
}

def Fusion_ConvBnReluOp : Fusion_Op<"conv_bn_relu", [Pure]> {
  let summary = "fused conv + batch-norm(mul + add) + relu";
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       AnyRankedTensor:$scale,
                       AnyRankedTensor:$shift);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $weight `,` $scale `,` $shift attr-dict
    `:` type($input) `,` type($weight) `,` type($scale) `,` type($shift)
    `->` type($result)
  }];
}

#endif // FUSION_OPS
