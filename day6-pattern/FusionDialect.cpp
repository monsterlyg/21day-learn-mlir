//===- FusionDialect.cpp - Fusion dialect -----------------------*- C++ -*-===//
//
// 方言实现：定义 dialect 的 TypeID / 注册信息，并在 initialize() 里注册所有 op。
//
//===----------------------------------------------------------------------===//

#include "FusionDialect.h"
#include "FusionOps.h"

using namespace mlir;
using namespace mlir::fusion;

#include "FusionOpsDialect.cpp.inc"

void FusionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "FusionOps.cpp.inc"
      >();
}
