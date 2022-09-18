#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace triton {

// Type inference
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type,
                                 tensorType.getEncoding());
  return i1Type;
}

static Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i32Type,
                                 tensorType.getEncoding());
  return i32Type;
}

static Type getPointerTypeFromTensor(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    Type elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    PointerType ptrType = PointerType::get(elementType, 1);
    return RankedTensorType::get(shape, ptrType, tensorType.getEncoding());
  }
  return Type();
}

// Parser & printer for assembly forms
ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type resultTypes[1];
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(resultTypes[0]))
    return failure();

  result.addTypes(resultTypes);

  SmallVector<Type> operandTypes;
  operandTypes.push_back(getPointerTypeFromTensor(resultTypes[0])); // ptr
  if (allOperands.size() >= 2)
    operandTypes.push_back(getI1SameShape(resultTypes[0])); // mask
  if (allOperands.size() >= 3)
    operandTypes.push_back(resultTypes[0]); // other

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void LoadOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer << getOperation()->getOperands();
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  printer << " : ";
  printer.printStrippedAttrOrType(getResult().getType());
}

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type valueType;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(valueType))
    return failure();

  SmallVector<Type> operandTypes;
  operandTypes.push_back(getPointerTypeFromTensor(valueType)); // ptr
  operandTypes.push_back(valueType);                           // value
  if (allOperands.size() >= 3)
    operandTypes.push_back(getI1SameShape(valueType)); // mask

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void StoreOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer << getOperation()->getOperands();
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  printer << " : ";
  printer.printStrippedAttrOrType(getValue().getType());
}

} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"

// enum attribute definitions
#include "triton/Dialect/Triton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {

//-- StoreOp --
void StoreOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    ::mlir::Value ptr, ::mlir::Value value) {
  StoreOp::build(builder, state, ptr, value, mlir::Value());
}

//-- LoadOp --
void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mlir::Value(), mlir::Value(), cache, evict,
                isVolatile);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::Value mask,
                   ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, mlir::Value(), cache, evict,
                isVolatile);
}

void LoadOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                   ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value other,
                   ::mlir::triton::CacheModifier cache,
                   ::mlir::triton::EvictionPolicy evict, bool isVolatile) {
  TensorType ptrType = ptr.getType().dyn_cast<TensorType>();
  Type elementType =
      ptrType.getElementType().dyn_cast<PointerType>().getPointeeType();
  auto shape = ptrType.getShape();
  Type resultType = RankedTensorType::get(shape, elementType);
  state.addOperands(ptr);
  if (mask) {
    state.addOperands(mask);
    if (other) {
      state.addOperands(other);
    }
  }
  state.addAttribute(
      getCacheAttrName(state.name),
      ::mlir::triton::CacheModifierAttr::get(builder.getContext(), cache));
  state.addAttribute(
      getEvictAttrName(state.name),
      ::mlir::triton::EvictionPolicyAttr::get(builder.getContext(), evict));
  state.addAttribute(getIsVolatileAttrName(state.name),
                     builder.getBoolAttr(isVolatile));
  state.addTypes({resultType});
}

//-- DotOp --

//-- SplatOp --
OpFoldResult SplatOp::fold(ArrayRef<Attribute> operands) {
  auto constOperand = getSrc().getDefiningOp<arith::ConstantOp>();
  if (!constOperand)
    return {};

  auto shapedType = getType().cast<ShapedType>();
  auto ret = SplatElementsAttr::get(
      shapedType, ArrayRef<Attribute>{constOperand.getValue()});
  return ret;
}

//-- BroadcastOp --
OpFoldResult BroadcastOp::fold(ArrayRef<Attribute> operands) {
  auto constOperand = getSrc().getDefiningOp<arith::ConstantOp>();
  if (!constOperand)
    return {};

  auto shapedType = getType().cast<ShapedType>();
  auto value = constOperand.getValue();
  if (auto denseElemsAttr = value.dyn_cast<DenseElementsAttr>()) {
    if (!denseElemsAttr.isSplat())
      return {};
    return SplatElementsAttr::get(shapedType,
                                  denseElemsAttr.getSplatValue<Attribute>());
  } else if (value.getType().isIntOrIndexOrFloat()) {
    return SplatElementsAttr::get(shapedType, value);
  } else {
    return {};
  }
}

} // namespace triton
} // namespace mlir
