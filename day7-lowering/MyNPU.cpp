// attr 定义里用到 FieldParser / TypeSwitch / AsmParser 等，需要这两个头
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "MyNPU.h"
#include "MyNPUDialect.cpp.inc"

#define GET_OP_CLASSES
#include "MyNPUOps.cpp.inc"


void mynpu::MyNPUDialect::initialize() {
  // 注册所有 op：GET_OP_LIST 展开成 "::mynpu::ReluOp" 列表
  addOperations<
#define GET_OP_LIST
#include "MyNPUOps.cpp.inc"
  >();
}