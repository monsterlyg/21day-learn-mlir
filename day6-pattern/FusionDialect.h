//===- FusionDialect.h - Fusion dialect -------------------------*- C++ -*-===//
//
// 声明 fusion 方言类。真正的类定义由 TableGen 生成在
// "FusionOpsDialect.h.inc" 里（构建时生成）。
//
//===----------------------------------------------------------------------===//

#ifndef FUSION_FUSIONDIALECT_H
#define FUSION_FUSIONDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

#include "FusionOpsDialect.h.inc"

#endif // FUSION_FUSIONDIALECT_H
