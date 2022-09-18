#ifndef TRITON_DIALECT_TRITON_IR_DIALECT_H_
#define TRITON_DIALECT_TRITON_IR_DIALECT_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.h.inc"

#endif // TRITON_IR_DIALECT_H_
