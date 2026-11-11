// DialectImplementation.h: attr 定义里用到 FieldParser / TypeSwitch / AsmParser
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "MyNPU.h"
#include "MyNPUOps.h"
#include "MyNPUCostOpInterface.cpp.inc"
#include "MyNPUDialect.cpp.inc"

using namespace mlir;

void mynpu::MyNPUDialect::initialize() {
  // 注册所有 op：GET_OP_LIST 展开成 "::mynpu::ReluOp" 列表
  addOperations<
#define GET_OP_LIST
#include "MyNPUOps.cpp.inc"
  >();
}
