//===- MyNPUOps.h - MyNPU dialect ops -----------------------*- C++ -*-===//
//
// Op 类声明（ReluOp / ConvOp / AddfOp），由 TableGen 生成。
//
//===----------------------------------------------------------------------===//

#ifndef MYNPU_OPS_H
#define MYNPU_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

#include "MyNPU.h"

#define GET_OP_CLASSES
#include "MyNPUOps.h.inc"

#endif // MYNPU_OPS_H
