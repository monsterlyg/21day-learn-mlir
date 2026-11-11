//===- FusionPassesPlugin.cpp - plugin entry points --------------*- C++ -*-===//
//
// 两个插件入口（同官方 standalone 例子）：
//   - mlirGetDialectPluginInfo：注册 fusion 方言 + pass。
//     用 `mlir-opt --load-dialect-plugin` 加载（pass 会创建 fusion.* op，
//     必须把方言注册进 context）。
//   - mlirGetPassPluginInfo：只注册 pass。用 `--load-pass-plugin` 加载；
//     fusion 方言会由 pass 的 getDependentDialects 按需加载。
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "FusionDialect.h"
#include "FusionPasses.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

// 方言插件入口
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Fusion", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<fusion::FusionDialect>();
            registerFusionPasses();
          }};
}

// pass 插件入口
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "FusionPasses", LLVM_VERSION_STRING,
          []() { registerFusionPasses(); }};
}
