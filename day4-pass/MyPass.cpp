#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <map>


using namespace mlir;

namespace {
struct StatisticPass : public PassWrapper<StatisticPass, 
        OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        StatisticPass
    )

    StringRef getArgument() const final { return "statistic-pass"; }
    StringRef getDescription() const final {
        return "Print statistic of all operations";
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        std::map<std::string,int> counter;

        module.walk([&](Operation *op) {
            auto opName = op->getName().getStringRef().str();
            counter[opName]++;
        });
        for (auto &[opName, count] : counter) {
            llvm::outs() << opName << " " << count << "\n";
        }
    }
};


struct ConvAddPass : public PassWrapper<ConvAddPass, 
        OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        ConvAddPass
    )

    StringRef getArgument() const final { return "conv-add-pass"; }
    StringRef getDescription() const final {
        return "Find Conv-Add operations";
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        module.walk([&](arith::AddFOp addOp) {
            for (Value operand : addOp.getOperands()) {
                Operation *defOp = operand.getDefiningOp();
                if (!defOp)
                    continue; // block argument / external value, no defining op

                if (isa<arith::MulFOp>(defOp)) {
                    llvm::outs() << "Conv-Add found\n";
                    llvm::outs() << "arith.mulf -> arith.addf\n";
                }
            }

        });
    }
};

} // end anonymous namespace

void registerMyPass(){
    PassRegistration<StatisticPass>();
    PassRegistration<ConvAddPass>();
}

//  mlir-opt --load-pass-plugin ./libMyPass.so --pass-pipeline='builtin.module(statistic-pass)' ../resblk.dummy.mlir