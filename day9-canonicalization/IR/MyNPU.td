#ifndef MYNPU_DIALECT
#define MYNPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "IR/MyNPUCostOpInterface.td"

def MyNPU_Dialect : Dialect {
  let name = "mynpu";
  let cppNamespace = "::mynpu";
}

include "IR/MyNPUOps.td"


#endif // MYNPU_DIALECT
