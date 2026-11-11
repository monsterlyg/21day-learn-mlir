//===- FusionOps.cpp - Fusion dialect ops -----------------------*- C++ -*-===//
//
// op 的构建 / parse / print / verify 实现（由 TableGen 生成）。
//
//===----------------------------------------------------------------------===//

#include "FusionOps.h"
#include "FusionDialect.h"

// 生成的 .cpp.inc 里会用到 OpBuilder / ImplicitLocOpBuilder / Builder 的完整定义
// （构造 build 方法），需要显式包含 Builders.h。
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "FusionOps.cpp.inc"
