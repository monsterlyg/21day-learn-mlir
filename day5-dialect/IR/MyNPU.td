#ifndef MYNPU_DIALECT
#define MYNPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"


def MyNPU_Dialect : Dialect {
  let name = "mynpu";
  let cppNamespace = "::mynpu";

  // 让 tablegen 自动生成 parseAttribute/printAttribute，
  // 分发到各 AttrDef 的 parse/print，否则 #mynpu.quant<...> 无法解析。
  let useDefaultAttributePrinterParser = 1;
}

include "MyNPUAttrs.td"
include "MyNPUOps.td"


#endif // MYNPU_DIALECT