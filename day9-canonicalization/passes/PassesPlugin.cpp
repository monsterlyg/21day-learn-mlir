

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

#include "IR/MyNPU.h"
#include "PassesPlugin.h"

using namespace mlir;

// Dialect plugin entry point - 注册方言（解析 MLIR 时使用）
// 注意：这里只注册 Dialect，不注册 Pass
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
    return {MLIR_PLUGIN_API_VERSION, "MyNPU", LLVM_VERSION_STRING,
            [](::mlir::DialectRegistry *registry) {
                registry->insert<mynpu::MyNPUDialect>();
                // Pass 注册在 mlirGetPassPluginInfo() 中进行
            }};
}

// Pass plugin entry point - 注册 Pass（运行 Pass 时使用）
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {MLIR_PLUGIN_API_VERSION, "MyNPUPasses", LLVM_VERSION_STRING,
            []() {
                registerMyNPULoweringPasses();
                registerMyNPUShapeInferencePass();
                registerMyNPUCostAnalysisPass();
            }};
}
