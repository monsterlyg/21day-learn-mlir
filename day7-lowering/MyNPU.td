#ifndef MYNPU_DIALECT
#define MYNPU_DIALECT

include "mlir/IR/OpBase.td"


def MyNPU_Dialect : Dialect {
  let name = "mynpu";
  let cppNamespace = "::mynpu";
}

include "MyNPUOps.td"


#endif // MYNPU_DIALECT