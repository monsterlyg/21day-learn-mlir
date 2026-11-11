// attr 定义里用到 FieldParser / TypeSwitch / AsmParser 等，需要这两个头
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "MyNPU.h"
#include "MyNPUDialect.cpp.inc"

// 1.生成：ReluOp 的 build / verify / 类型约束检查等方法定义
#define GET_OP_CLASSES
#include "MyNPUOps.cpp.inc"

// 2.addAttributes<QuantAttr>() 需要 QuantAttr 的 storage 完整定义，
// 所以把 attrdef 定义（含 quantAttrStorage 结构体 + 方法实现）放在这里。
#define GET_ATTRDEF_CLASSES
#include "MyNPUAttrs.cpp.inc"


// 3.注册 dialect / op / attr
#include "mlir/Tools/Plugins/DialectPlugin.h"

void mynpu::MyNPUDialect::initialize() {
  // 注册所有 op：GET_OP_LIST 展开成 "::mynpu::ReluOp" 列表
  addOperations<
#define GET_OP_LIST
#include "MyNPUOps.cpp.inc"
  >();

  // 注册自定义属性：GET_ATTRDEF_LIST 展开成 "::mynpu::QuantAttr" 列表
  addAttributes<
#define GET_ATTRDEF_LIST
#include "MyNPUAttrs.cpp.inc"
  >();
}

// 4. dialect 插件入口：让 mlir-opt 通过 --load-dialect-plugin 加载本库时，
// 把 MyNPUDialect 注册进 DialectRegistry。
extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MyNPU", "v0.1",
          [](::mlir::DialectRegistry *registry) {
            registry->insert<mynpu::MyNPUDialect>();
          }};
}