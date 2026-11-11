#include "IR/MyNPU.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mynpu {

struct MyNPUCostAnalysisPass
    : public PassWrapper<MyNPUCostAnalysisPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyNPUCostAnalysisPass)

    StringRef getArgument() const final { return "mynpu-cost-analysis"; }
    StringRef getDescription() const final {
        return "Sum compute costs reported by MyNPUCostOpInterface";
    }

    void runOnOperation() override {
        int64_t totalCost = 0;
        getOperation().walk([&](Operation *op) {
        if (auto costOp = dyn_cast<MyNPUCostOpInterface>(op))
            totalCost += costOp.getComputeCost();
        });
        getOperation().emitRemark() << "Total Compute Cost = " << totalCost;
    }
};

} // namespace mynpu

void registerMyNPUCostAnalysisPass() {
    PassRegistration<mynpu::MyNPUCostAnalysisPass>();
}
