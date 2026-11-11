//===- FusionOps.h - Fusion dialect ops -------------------------*- C++ -*-===//
//
// 声明 fusion 方言的两个 op 类（ConvAddReluOp / ConvBnReluOp），由 TableGen 生成。
//
//===----------------------------------------------------------------------===//

#ifndef FUSION_FUSIONOPS_H
#define FUSION_FUSIONOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "FusionOps.h.inc"

#endif // FUSION_FUSIONOPS_H
