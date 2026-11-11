#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"



int main(int argc, char **argv) {
    if (argc < 2) {
        llvm::errs() << "Usage: " << argv[0] << " <mlir-file>\n";
        return 1;
    }
    mlir::MLIRContext context;
    context.loadDialect<
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect>();

    llvm::SourceMgr sourceMgr;

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        argv[1], sourceMgr, &context);

    if (!module) {
        llvm::errs() << "Failed to parse MLIR\n";
        return 1;
    }

    module->walk([](mlir::Operation *op) {
        if (op->getName().getDialectNamespace() == "arith") {
            auto addf = llvm::dyn_cast<mlir::arith::AddFOp>(op);
            if (!addf) return;
            llvm::outs() << "Operation: " << op->getName() << "\n";
            llvm::outs() << "Dialect: " << op->getName().getDialectNamespace() << "\n";
            llvm::outs() << "Operands: " << "\n";
            unsigned int oprs = op->getNumOperands();
            
            for(int i = 0; i < oprs; ++i){
                llvm::outs() << "  operand " << i << "\n";
                mlir::Value operand = op->getOperand(i);
                llvm::outs() << "    Value: ";
                operand.print(llvm::outs());
                llvm::outs() << "    Type: " << operand.getType() << "\n";
            }

            unsigned int ret_num = op->getNumResults();
            llvm::outs() << "Results: " << "\n";
            for(int i = 0; i < ret_num; ++i){
                mlir::Value result = op->getResult(i);
                llvm::outs() << "  result " << i << "\n";
                llvm::outs() << "    Value: ";
                result.print(llvm::outs());
                llvm::outs() << "    Type: " << result.getType() << "\n";
            }
        } else {
            llvm::outs() << "Dialect: " << op->getName().getDialectNamespace() << "\n";
        }
    });

}