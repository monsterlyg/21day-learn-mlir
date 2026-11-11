#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
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

    // mlir::Operation *op = module.getOperation();
    // op->dump();
    module->walk([](mlir::Operation *op) {
        llvm::StringRef op_name = op->getName().getStringRef();
        llvm::outs() << "operation = " << op_name << "\n";
        mlir::ResultRange results = op->getResults();
        for (mlir::Value result : results) {
            llvm::outs() << "result type = ";
            result.getType().print(llvm::outs());
            llvm::outs() << "\n";
        }

        mlir::ArrayRef<mlir::NamedAttribute> attrs = op->getAttrs();
        for (const mlir::NamedAttribute &attr : attrs) {
            llvm::outs() << "attr = ";
            attr.getName().print(llvm::outs());
            llvm::outs() << " : ";
            attr.getValue().print(llvm::outs());
            llvm::outs() << "\n";
        }
        if(attrs.empty()) {
            llvm::outs() << "attrs.empty()\n";
        }

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            mlir::Value value = op->getOperand(i);

            llvm::outs() << "operand " << i
                        << ", type = ";

            value.getType().print(llvm::outs());

            llvm::outs() << "\n";
        }

        // 打印location
        mlir::Location loc = op->getLoc();
        llvm::outs() << "location = ";
        loc.print(llvm::outs());
        llvm::outs() << "\n";
        
        // 打印region信息
        unsigned region_count = op->getNumRegions();
        if (region_count > 0) {
            llvm::outs() << "regions (" << region_count << "):\n";
            for (mlir::Region &region : op->getRegions()) {
                llvm::outs() << "  region:\n";
                for (mlir::Block &block : region.getBlocks()) {
                    llvm::outs() << "    block:\n";
                    for (mlir::Operation &inner_op : block.getOperations()) {
                        llvm::outs() << "      " << inner_op.getName().getStringRef() << "\n";
                    }
                }
            }
        } else {
            llvm::outs() << "no regions\n";
        }
        llvm::outs() << "\n";
    });

    return 0;
}