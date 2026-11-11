#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "IR/MyNPU.h"
#include "PassesPlugin.h"
using namespace mlir;

namespace mynpu {

struct MyNPUShapeInferencePass
    : public mlir::PassWrapper<MyNPUShapeInferencePass,
                            mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyNPUShapeInferencePass)

    mlir::StringRef getArgument() const final { return "mynpu-shape-inference"; }
    mlir::StringRef getDescription() const final {
        return "Infer shapes for MyNPU ops";
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        module.walk([&](Operation *op) {
            if (auto inferOp = dyn_cast<InferTypeOpInterface>(op)) {
                SmallVector<Type, 4> inferredTypes;
                if (succeeded(inferOp.inferReturnTypes(
                        op->getContext(),
                        std::nullopt,
                        op->getOperands(),
                        op->getAttrDictionary(),
                        op->getPropertiesStorage(),
                        op->getRegions(),
                        inferredTypes))) 
                {
                    if (inferredTypes.size() == op->getNumResults()) {
                        for (size_t i = 0; i < inferredTypes.size(); ++i) {
                            op->getResult(i).setType(inferredTypes[i]);
                        }
                    }
                }
            }
        });
    }
};

} // namespace mynpu

void registerMyNPUShapeInferencePass() {
    mlir::PassRegistration<mynpu::MyNPUShapeInferencePass>();
}