#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

#include "MyNPUDialect.h.inc"
// 属性声明：QuantAttr（ConvOp 的 build/accessor 会引用它，必须先于 op 声明）
#define GET_ATTRDEF_CLASSES
#include "MyNPUAttrs.h.inc"

#define GET_OP_CLASSES
#include "MyNPUOps.h.inc"