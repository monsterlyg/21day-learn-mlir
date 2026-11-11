#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

int main1(int argc, char **argv) {
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

        // 打印region信息
        unsigned region_count = op->getNumRegions();
        if (region_count > 0) {
            llvm::outs() << "regions (" << region_count << "):\n";
            for (unsigned i = 0; i < region_count; ++i) {
                llvm::outs() << "  region " << i << ":\n";
                mlir::Region &region = op->getRegion(i);
                
                for (mlir::Block &block : region.getBlocks()) {
                    llvm::outs() << "    block:\n";
                    for(mlir::BlockArgument &arg : block.getArguments()) {
                        llvm::outs() << "      BlockArgument: ";
                        arg.print(llvm::outs());
                        llvm::outs() << "\n";
                    }
                    for (mlir::Operation &inner_op : block.getOperations()) {
                        llvm::outs() << "      " << inner_op.getName().getStringRef() << "\n";

                        for(unsigned j = 0; j < inner_op.getNumOperands(); ++j) {
                            mlir::Value value = inner_op.getOperand(j);

                            llvm::outs() << "        operand " << j << ", type = ";
                            value.getType().print(llvm::outs());
                            llvm::outs() << "\n";

                            if (mlir::Operation *def_op = value.getDefiningOp()) {
                                llvm::outs() << "          defining op: " << def_op->getName().getStringRef() << "\n";
                            }
                            for (mlir::Operation *user_op : value.getUsers()) {
                                llvm::outs() << "          user op: " << user_op->getName().getStringRef() << "\n";
                            }
                        }

                        for (unsigned j = 0; j < inner_op.getNumResults(); ++j) {
                            mlir::Type type = inner_op.getResult(j).getType();
                            llvm::outs() << "        result " << j << ", type = ";
                            type.print(llvm::outs());
                            llvm::outs() << "\n";
                        }
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

    if (module) {
    // 查找名为 "main" 的函数（或者遍历所有函数）
        for (auto func : module->getOps<mlir::func::FuncOp>()) {
            if (func.getName() == "main") {
                // 使用 walk 找到所有 return 操作
                func.walk([&](mlir::func::ReturnOp returnOp) {
                    if (returnOp.getNumOperands() > 0) {
                        mlir::Value retVal = returnOp.getOperand(0);
                        // 打印或使用这个值
                        llvm::outs() << "Return value: ";
                        retVal.dump();
                        // 注意：如果函数有多个 return，这里会多次触发

                        mlir::Operation *defOp = retVal.getDefiningOp();
                        llvm::outs() << "Defining operation: " << defOp->getName().getStringRef() << "\n";
                        unsigned operandCount = defOp->getNumOperands();
                        for (unsigned i = 0; i < operandCount; ++i) {
                            llvm::outs() << "Operand " << i << ": ";
                            mlir::Value operand = defOp->getOperand(i);
                            operand.print(llvm::outs());
                            llvm::outs() << "\n";
                            if (mlir::Operation *operandDefOp = operand.getDefiningOp()) {
                                operandDefOp->print(llvm::outs());
                            }
                            llvm::outs() << "\n";
                        }
                    }
                });
                break; // 只处理第一个匹配的函数
            }
        }
    }

}