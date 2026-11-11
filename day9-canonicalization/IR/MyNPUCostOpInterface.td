#ifndef MYNPU_COST_OP_INTERFACE
#define MYNPU_COST_OP_INTERFACE

include "mlir/IR/OpBase.td"

def MyNPUCostOpInterface : OpInterface<"MyNPUCostOpInterface"> {
    let cppNamespace = "::mynpu";
    let description = [{
        Reports an operation's estimated compute cost in floating-point operations.
    }];
    let methods = [
        InterfaceMethod<
        /*desc=*/[{Return the estimated compute cost in FLOPs.}],
        /*retTy=*/"int64_t",
        /*methodName=*/"getComputeCost"
        >
    ];
}

#endif // MYNPU_COST_OP_INTERFACE