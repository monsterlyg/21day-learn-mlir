#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

#include "MyNPUDialect.h.inc"

#define GET_OP_CLASSES
#include "MyNPUOps.h.inc"