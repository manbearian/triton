//===- TritonToAffine.cpp - Triton to standard dialects conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "triton/Conversion/TritonToAffine/TritonToAffine.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//


// Consumed by addptr -> subview
// This is a bit of a hack... this could probably be an earlier analysis pass
static bool isAddressComputationOp(Operation *op) {
  assert(op->getResults().size() == 1);
  auto opType = op->getResultTypes()[0].template cast<TensorType>();
  if (opType.getElementType().isa<triton::PointerType>()) {
    return true;
  }

  // SUPER HACK -- fix with some real analysis?!?!
  return llvm::all_of(op->getUsers(), [op](Operation *user) {
    if (isa<triton::AddPtrOp>(user))
      return true;
    if (auto load = dyn_cast<triton::LoadOp>(user))
      if (op == load.getPtr().getDefiningOp() ||
          op == load.getMask().getDefiningOp())
        return true;
    if (auto store = dyn_cast<triton::StoreOp>(user))
      if (op == store.getPtr().getDefiningOp() ||
          op == store.getMask().getDefiningOp())
        return true;
    if (isa<arith::AddIOp, arith::CmpIOp>(user))
      return isAddressComputationOp(user);
    return false;
  });
}

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (isAddressComputationOp(op)) {
      // Consumed by addptr -> subview
      rewriter.eraseOp(op);
      return success();
    }

    auto opType = op.getType().cast<TensorType>();
    auto type = MemRefType::get(opType.getShape(), opType.getElementType());
    auto res = rewriter.create<memref::AllocOp>(op.getLoc(), type);

    auto lbs = SmallVector<int64_t>(type.getShape().size(), 0);
    auto ubs = type.getShape();
    auto steps = SmallVector<int64_t>(type.getShape().size(), 1);
    buildAffineLoopNest(rewriter, op.getLoc(), lbs, ubs, steps,
                        [&](OpBuilder &b, Location loc, ValueRange ivs) {
                          b.create<AffineStoreOp>(loc, op.getSrc(), res, ivs);
                        });
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (isAddressComputationOp(op)) {
      // Consumed by addptr -> subview
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isAddressComputationOp(op)) {
      // Consumed by addptr -> subview
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto range = llvm::seq(op.getStart(), op.getEnd());
    auto array = std::vector<int64_t>(range.begin(), range.end());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, op.getType(), rewriter.getIndexVectorAttr(array));
    return success();
  }
};

// struct UndefConverter : public OpConversionPattern<triton::UndefOp> {
//   using OpConversionPattern<triton::UndefOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(triton::UndefOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, op.getType());
//     return success();
//   }
// };

struct AddPtrConverter
    : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  // clang-format off
  // %72 = arith.muli %14, %c128 : index
  // %73 = "triton.make_range"() {start = 0 : index, end = 128 : index} : () -> (tensor<128xindex>)
  // %74 = "triton.splat"(%72) : (index) -> (tensor<128xindex>)
  // %75 = arith.addi %74, %73 : tensor<128xindex>
  // clang-format on
  Value getOffset(Value v, int64_t size) const {
    auto addOp = v.getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return nullptr;

    auto splatIndex = addOp.getLhs().getDefiningOp<triton::SplatOp>();
    auto rangeOp = addOp.getRhs().getDefiningOp<triton::MakeRangeOp>();
    if (!splatIndex || !rangeOp) {
      splatIndex = addOp.getRhs().getDefiningOp<triton::SplatOp>();
      rangeOp = addOp.getLhs().getDefiningOp<triton::MakeRangeOp>();
      if (!splatIndex || !rangeOp)
        return nullptr;
    }

    auto scaleOp = splatIndex.getSrc().getDefiningOp<arith::MulIOp>();
    if (!scaleOp) {
      return nullptr;
    }

    auto scale = scaleOp.getRhs().getDefiningOp<arith::ConstantOp>();
    auto offset = scaleOp.getLhs();

    if (!scale ||
        (scale.getValue().template cast<IntegerAttr>().getInt() != size)) {
      return nullptr;
    }

    if (rangeOp.getStart() != 0 || rangeOp.getEnd() != size)
      return nullptr;

    return offset;
  }

  static ArrayRef<int64_t> getShape(triton::AddPtrOp op) {
    // for block-args the "vector of index" type will already have been
    // replaced with a memref
    if (auto type = op.getPtr().getType().dyn_cast<MemRefType>()) {
      return type.getShape();
    }
    return op.getPtr().getType().cast<ShapedType>().getShape();
  }

  // clang-format off
  // %1 = arith.muli %0, %c1024 : index
  // %2 = "triton.make_range"() {start = 0 : index, end = 1024 : index}  : () -> (tensor<1024xindex>)
  // %3 = "triton.splat"(%1) : (index) -> (tensor<1024xindex>)
  // %4 = arith.addi %3, %2 : tensor<1024xindex>
  // %7 = "triton.splat"(%arg0) : (memref<bf16>) -> (tensor<1024xindex>)
  // %8 = "triton.addptr"(%7, %4) { ptrType = bf16 }  : (tensor<1024xindex>, tensor<1024xindex>) -> tensor<1024xindex>
  // clang-format on
  LogicalResult
  trySimplePointerAndIndex(triton::AddPtrOp op,
                           ConversionPatternRewriter &rewriter) const {
    auto shape = getShape(op);
    if (shape.size() != 1)
      return failure();

    auto splatPtr = op.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splatPtr)
      return failure();

    auto pointer = splatPtr.getSrc();
    auto offset = getOffset(op.getOffset(), shape[0]);

    SmallVector<OpFoldResult> offsets = {offset};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(shape[0])};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
    auto origType = pointer.getType().cast<UnrankedMemRefType>();
    auto updateType = MemRefType::get({-1}, origType.getElementType());
    auto cast =
        rewriter.create<memref::CastOp>(op.getLoc(), updateType, pointer);
    auto subType =
        memref::SubViewOp::inferResultType(updateType, offsets, sizes, strides)
            .cast<MemRefType>();
    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), subType, cast, offsets, sizes, strides);
    rewriter.replaceOp(op, subview.getResult());

    return success();
  }

  // clang-format off
  // Sequence A -- fixed index
  // %25 = "triton.make_range"() {start = 0 : index, end = 32 : index} : () -> (tensor<32xindex>) // BLOCK_SIZE_K
  // %37 = "triton.reshape"(%25) : (tensor<32xindex>) -> (tensor<32x1xindex>)
  // %38 = "triton.splat"(%arg8) : (index) -> (tensor<32x1xindex>) // stride_bk
  // %39 = arith.muli %37, %38 : tensor<32x1xindex>
  // %43 = "triton.broadcast"(%39) : (tensor<32x1xindex>) -> (tensor<32x256xindex>)
  //  or
  // Sequence B -- scaled index
  // %21 = arith.muli %16, %c256 : index // BLOCK_SIZE_N
  // %22 = "triton.make_range"() {start = 0 : index, end = 256 : index} : () -> (tensor<256xindex>)
  // %23 = "triton.splat"(%21) : (index) -> (tensor<256xindex>)
  // %24 = arith.addi  %23, %22 : tensor<256xindex>
  // %40 = "triton.reshape"(%24) : (tensor<256xindex>) -> (tensor<1x256xindex>)
  // %41 = "triton.splat"(%c1) : (index) -> (tensor<1x256xindex>)
  // %42 = arith.muli %40, %41 : tensor<1x256xindex>
  // %44 = "triton.broadcast"(%42) : (tensor<1x256xindex>) -> (tensor<32x256xindex>)
  // clang-format on
  struct DimInfo {
    OpFoldResult offset;
    OpFoldResult size;
    OpFoldResult stride;
    DimInfo() : offset(nullptr), size(nullptr), stride(nullptr){};
    DimInfo(OpFoldResult offset, OpFoldResult size, OpFoldResult stride)
        : offset(offset), size(size), stride(stride) {}
  };

  DimInfo computeDim(Value v, ConversionPatternRewriter &b) const {
    auto shape = v.getType().cast<ShapedType>().getShape();
    if (shape.size() != 2)
      return {};
    auto size = (shape[0] == 1) ? shape[1] : shape[0];
    if (size <= 1)
      return {};

    auto mulOp = v.getDefiningOp<arith::MulIOp>();
    if (!mulOp)
      return {};

    auto reshape = mulOp.getLhs().getDefiningOp<triton::ExpandDimsOp>();
    auto strideSplat = mulOp.getRhs().getDefiningOp<triton::SplatOp>();
    if (!reshape || !strideSplat) {
      reshape = mulOp.getRhs().getDefiningOp<triton::ExpandDimsOp>();
      strideSplat = mulOp.getLhs().getDefiningOp<triton::SplatOp>();
      if (!reshape || !strideSplat)
        return {};
    }

    OpFoldResult stride = strideSplat.getSrc();
    if (auto cStride = stride.get<Value>().getDefiningOp<arith::ConstantOp>()) {
      stride = cStride.getValue();
    }

    // fixed index
    if (auto rangeOp = reshape.getSrc().getDefiningOp<triton::MakeRangeOp>()) {
      if (rangeOp.getStart() != 0 || rangeOp.getEnd() != size)
        return {};
      return {b.getIndexAttr(0), b.getIndexAttr(size), stride};
    }

    // variable index
    auto offset = getOffset(reshape.getSrc(), size);
    if (!offset)
      return {};

    return {offset, b.getIndexAttr(size), stride};
  }

  // clang-format off
  // Compute "a_ptrs"
  // // t1 = offs_am[:,None] * stride_am
  // %26 = "triton.reshape"(%20) : (tensor<128xindex>) -> (tensor<128x1xindex>)
  // %27 = "triton.splat"(%arg6) : (index) -> (tensor<128x1xindex>) // stride_am
  // %28 = arith.muli %26, %27 : tensor<128x1xindex>
  // //  t2 = offs_k[None,:] * stride_ak
  // %29 = "triton.reshape"(%25) : (tensor<32xindex>) -> (tensor<1x32xindex>)
  // %30 = "triton.splat"(%c1) : (index) -> (tensor<1x32xindex>)
  // %31 = arith.muli %29, %30 : tensor<1x32xindex>
  // //  t3 = t1 + t2
  // %32 = "triton.broadcast"(%28) : (tensor<128x1xindex>) -> (tensor<128x32xindex>)
  // %33 = "triton.broadcast"(%31) : (tensor<1x32xindex>) -> (tensor<128x32xindex>)
  // %34 = arith.addi %32, %33: tensor<128x32xindex> // a_ptrs = a_ptr + t3
  // %35 = "triton.splat"(%arg0) : (memref<?xbf16>) -> (tensor<128x32xindex>) // a_ptr
  // %36 = "triton.addptr"(%35, %34) {ptrType = bf16 } : (tensor<128x32xindex>, tensor<128x32xindex>) -> tensor<128x32xindex>
  // clang-format on
  LogicalResult try2dMatrixType1(triton::AddPtrOp op,
                                 ConversionPatternRewriter &rewriter) const {
    auto shape = getShape(op);
    if (shape.size() != 2)
      return failure();

    auto splatPtr = op.getPtr().getDefiningOp<triton::SplatOp>();
    if (!splatPtr)
      return failure();

    auto pointer = splatPtr.getSrc();

    auto addOp = op.getOffset().getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    auto broadcast0 = addOp.getLhs().getDefiningOp<triton::BroadcastOp>();
    auto broadcast1 = addOp.getRhs().getDefiningOp<triton::BroadcastOp>();
    if (!broadcast0 || !broadcast1)
      return failure();

    auto dim0 = computeDim(broadcast0.getSrc(), rewriter);
    auto dim1 = computeDim(broadcast1.getSrc(), rewriter);
    if (!dim0.offset || !dim1.offset)
      return failure();

    SmallVector<OpFoldResult> offsets = {dim0.offset, dim1.offset};
    SmallVector<OpFoldResult> sizes = {dim0.size, dim1.size};
    SmallVector<OpFoldResult> strides = {dim0.stride, dim1.stride};
    auto origType = pointer.getType().cast<UnrankedMemRefType>();
    auto updateType = MemRefType::get({-1, -1}, origType.getElementType());
    auto cast =
        rewriter.create<memref::CastOp>(op.getLoc(), updateType, pointer);
    auto subType =
        memref::SubViewOp::inferResultType(updateType, offsets, sizes, strides)
            .cast<MemRefType>();
    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), subType, cast, offsets, sizes, strides);
    rewriter.replaceOp(op, subview.getResult());

    return success();
  }

  // clang-format off
  // %72 = arith.muli %14, %c128 : index
  // %73 = "triton.make_range"() {start = 0 : index, end = 128 : index}  : () -> (tensor<128xindex>)
  // %74 = "triton.splat"(%72) : (index) -> (tensor<128xindex>)
  // %75 = arith.addi %74, %73 : tensor<128xindex>
  //
  // %76 = arith.muli %16, %c256 : index
  // %77 = "triton.make_range"() {start = 0 : index, end = 256 : index}  : () -> (tensor<256xindex>)
  // %78 = "triton.splat"(%7) : (index) -> (tensor<256xindex>)
  // %79 = arith.addi %78, %77 : tensor<256xindex>
  //
  // %80 = "triton.reshape"(%75) : (tensor<128xindex>) -> (tensor<128x1xindex>)
  // %81 = "triton.splat"(%arg10) : (index) -> (tensor<128x1xindex>) // stride_cm
  // %82 = arith.muli %81, %80 : tensor<128x1xindex>
  // %83 = "triton.splat"(%arg2) : (memref<?xbf16>) -> (tensor<128x1xindex>) // c_ptr
  // %84 = "triton.getelementptr"(%83, %82) {ptrType = bf16 } : (tensor<128x1xindex>, tensor<128x1xindex>) -> tensor<128x1xindex>
  //
  // %85 = "triton.reshape"(%79) : (tensor<256xindex>) -> (tensor<1x256xindex>)
  // %86 = "triton.splat"(%c1) : (index) -> (tensor<1x256xindex>)
  // %87 = arith.muli %85, %86 : tensor<1x256xindex>
  // %88 = "triton.broadcast"(%84) : (tensor<128x1xindex>) -> (tensor<128x256xindex>)
  // %89 = "triton.broadcast"(%87) : (tensor<1x256xindex>) -> (tensor<128x256xindex>)
  // %90 = "triton.getelementptr"(%88, %89) {ptrType = bf16 } : (tensor<128x256xindex>, tensor<128x256xindex>) -> tensor<128x256xindex>
  // clang-format on
  LogicalResult try2dMatrixType2(triton::AddPtrOp op,
                                 ConversionPatternRewriter &rewriter) const {
    auto shape = getShape(op);
    if (shape.size() != 2)
      return failure();

    auto splatPtr = op.getPtr().getDefiningOp<triton::BroadcastOp>();
    auto splatOffset = op.getOffset().getDefiningOp<triton::BroadcastOp>();
    if (!splatPtr || !splatOffset)
      return failure();

    auto gep2 = splatPtr.getSrc().getDefiningOp<triton::AddPtrOp>();
    if (!gep2)
      return failure();

    auto splat2 = gep2.getPtr().getDefiningOp<triton::SplatOp>();
    auto dim0 = computeDim(gep2.getOffset(), rewriter);
    auto dim1 = computeDim(splatOffset.getSrc(), rewriter);

    if (!splat2 || !dim0.offset || !dim1.offset)
      return failure();

    SmallVector<OpFoldResult> offsets = {dim0.offset, dim1.offset};
    SmallVector<OpFoldResult> sizes = {dim0.size, dim1.size};
    SmallVector<OpFoldResult> strides = {dim0.stride, dim1.stride};
    auto pointer = splat2.getSrc();
    auto origType = pointer.getType().cast<UnrankedMemRefType>();
    auto updateType = MemRefType::get({-1, -1}, origType.getElementType());
    auto cast =
        rewriter.create<memref::CastOp>(op.getLoc(), updateType, pointer);
    auto subType =
        memref::SubViewOp::inferResultType(updateType, offsets, sizes, strides)
            .cast<MemRefType>();
    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), subType, cast, offsets, sizes, strides);
    rewriter.replaceOp(op, subview.getResult());

    return success();
  }

  LogicalResult tryLoopUpdate(triton::AddPtrOp op,
                              ConversionPatternRewriter &rewriter) const {
    auto shape = getShape(op);
    if (shape.size() != 2)
      return failure();

    auto blockArg = op.getPtr().dyn_cast<BlockArgument>();
    if (!blockArg)
      return failure();
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return failure();
    auto initV = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
    auto mappedV = rewriter.getRemappedValue(initV);
    auto initSubview = mappedV.getDefiningOp<memref::SubViewOp>();
    if (!initSubview)
      return failure();

    SmallVector<OpFoldResult> offsets = initSubview.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = initSubview.getMixedSizes();
    SmallVector<OpFoldResult> strides = initSubview.getMixedStrides();

    auto splat = op.getOffset().getDefiningOp<triton::SplatOp>();
    if (!splat)
      return failure();

    if (auto offset = splat.getSrc().getDefiningOp<arith::ConstantOp>()) {
      if (offset.getValue().cast<IntegerAttr>().getInt() != shape[1])
        return failure();
      offsets[1] = *forOp.getSingleInductionVar();
    } else if (auto scale = splat.getSrc().getDefiningOp<arith::MulIOp>()) {
      if (scale.getLhs() != strides[0].dyn_cast<Value>())
        return failure();
      auto index = scale.getRhs().getDefiningOp<arith::ConstantOp>();
      if (!index || index.getValue().cast<IntegerAttr>().getInt() != shape[0])
        return failure();
      offsets[0] = *forOp.getSingleInductionVar();
    } else {
      return failure();
    }

    auto subType = memref::SubViewOp::inferResultType(initSubview.getType(),
                                                      offsets, sizes, strides)
                       .cast<MemRefType>();

    auto subview = rewriter.create<memref::SubViewOp>(
        op->getLoc(), subType, initSubview.source(), offsets, sizes, strides);
    rewriter.replaceOp(op, subview.getResult());

    return success();
  }

  // if the user is a GEP handle when we process that GEP
  LogicalResult tryGepToGep(triton::AddPtrOp op,
                            ConversionPatternRewriter &rewriter) const {
    if (!op->hasOneUse())
      return failure();
    auto broadcast = dyn_cast<triton::BroadcastOp>(*op->getUsers().begin());
    if (!broadcast || !broadcast->hasOneUse())
      return failure();
    bool isGep = isa<triton::AddPtrOp>(*broadcast->getUsers().begin());
    if (!isGep)
      return failure();

    // GEP->GEP will be consumed later when processing the later GEP
    rewriter.eraseOp(op);
    rewriter.eraseOp(broadcast);
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (succeeded(trySimplePointerAndIndex(op, rewriter))) {
      return success();
    }

    if (succeeded(try2dMatrixType1(op, rewriter))) {
      return success();
    }

    if (succeeded(try2dMatrixType2(op, rewriter))) {
      return success();
    }

    if (succeeded(tryLoopUpdate(op, rewriter))) {
      return success();
    }

    if (succeeded(tryGepToGep(op, rewriter))) {
      return success();
    }

    return failure();
  }
};

struct LoadConverter
    : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = rewriter.getRemappedValue(op.getPtr());
    auto type = ptr.getType().cast<MemRefType>();
    auto alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get(type.getShape(), type.getElementType()));
    rewriter.create<memref::CopyOp>(op.getLoc(), ptr, alloc);
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

struct StoreConverter
    : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = rewriter.getRemappedValue(op.getPtr());
    auto val = rewriter.getRemappedValue(op.getValue());
    rewriter.create<memref::CopyOp>(op.getLoc(), val, ptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LoopConverter : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> newInitArgs;
    for (auto arg : op.getInitArgs()) {
      if (auto mappedV = rewriter.getRemappedValue(arg)) {
        if (mappedV.getType() != arg.getType()) {
          newInitArgs.push_back(mappedV);
          continue;
        }
      }
      newInitArgs.push_back(arg);
    }

    auto newOp = rewriter.create<scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        newInitArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          BlockAndValueMapping mapping;
          mapping.map(op.getInductionVar(), iv);
          mapping.map(op.getInitArgs(), newInitArgs);
          mapping.map(op.getRegionIterArgs(), args);
          for (auto &bodyOp : op.getLoopBody().getOps()) {
            b.clone(bodyOp, mapping);
          }
        });
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct YieldConverter : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Dummy Converter...?
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct DotConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto lhs = rewriter.getRemappedValue(op.getA());
    auto rhs = rewriter.getRemappedValue(op.getB());
    auto acc = op.getC();

    // @TODO: Support General `acc`
    auto accDef = acc.getDefiningOp<triton::SplatOp>();
    if (!accDef)
      return failure();
    auto val = accDef.getSrc().getDefiningOp<arith::ConstantOp>();
    if (!val || (val.getValue().cast<FloatAttr>().getValueAsDouble() != 0.))
      return failure();

    auto opType = op.getType().cast<MemRefType>();
    auto type = MemRefType::get(opType.getShape(), opType.getElementType());
    auto res = rewriter.create<memref::AllocOp>(op.getLoc(), type);

    rewriter.create<linalg::MatmulOp>(op.getLoc(), ValueRange{lhs, rhs},
                                      ValueRange{res});
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};

template <typename ArithOp>
struct ArithConverter : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (isAddressComputationOp(op)) {
      // Consumed by addptr -> subview
      rewriter.eraseOp(op);
      return success();
    }

    assert(op->getResults().size() == 1);
    auto opType = op->getResultTypes()[0].template cast<TensorType>();

    auto ins = SmallVector<Value>();
    if (failed(rewriter.getRemappedValues(op->getOperands(), ins))) {
      return failure();
    }
    auto type = MemRefType::get(opType.getShape(), opType.getElementType());
    auto res = rewriter.create<memref::AllocOp>(op.getLoc(), type);

    auto lbs = SmallVector<int64_t>(type.getShape().size(), 0);
    auto ubs = type.getShape();
    auto steps = SmallVector<int64_t>(type.getShape().size(), 1);
    buildAffineLoopNest(rewriter, op.getLoc(), lbs, ubs, steps,
                        [&](OpBuilder &b, Location loc, ValueRange ivs) {
                          SmallVector<Value> lds;
                          for (auto v : ins) {
                            lds.push_back(b.create<AffineLoadOp>(loc, v, ivs));
                          }
                          auto op = b.create<ArithOp>(loc, lds);
                          b.create<AffineStoreOp>(loc, op, res, ivs);
                        });
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};

// TODO: this very close to ArithConverter except for the create<ArithOp> call
// with C++17 support use if constexpr in ArithConverter to specialize
template<typename ArithOp>
struct NumericCastConverter : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getResults().size() == 1);
    auto opType = op->getResultTypes()[0].template cast<TensorType>();

    auto ins = SmallVector<Value>();
    if (failed(rewriter.getRemappedValues(op->getOperands(), ins))) {
      return failure();
    }
    assert(ins.size() == 1);

    auto type = MemRefType::get(opType.getShape(), opType.getElementType());
    auto res = rewriter.create<memref::AllocOp>(op.getLoc(), type);

    auto lbs = SmallVector<int64_t>(type.getShape().size(), 0);
    auto ubs = type.getShape();
    auto steps = SmallVector<int64_t>(type.getShape().size(), 1);
    buildAffineLoopNest(rewriter, op.getLoc(), lbs, ubs, steps,
                        [&](OpBuilder &b, Location loc, ValueRange ivs) {
                          auto ld = b.create<AffineLoadOp>(loc, ins[0], ivs);
                          ArrayRef<mlir::Type> resultTypes = {opType.getElementType()};
                          auto op = b.create<ArithOp>(loc, resultTypes, ld);
                          b.create<AffineStoreOp>(loc, op, res, ivs);
                        });
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};

template<arith::CmpFPredicate p>
struct CmpFConverter : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, arith::CmpFOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getResults().size() == 1);
    auto opType = op->getResultTypes()[0].template cast<TensorType>();

    auto ins = SmallVector<Value>();
    if (failed(rewriter.getRemappedValues(op->getOperands(), ins))) {
      return failure();
    }
    assert(ins.size() == 2);

    auto type = MemRefType::get(opType.getShape(), opType.getElementType());
    auto res = rewriter.create<memref::AllocOp>(op.getLoc(), type);

    auto lbs = SmallVector<int64_t>(type.getShape().size(), 0);
    auto ubs = type.getShape();
    auto steps = SmallVector<int64_t>(type.getShape().size(), 1);
    buildAffineLoopNest(rewriter, op.getLoc(), lbs, ubs, steps,
                        [&](OpBuilder &b, Location loc, ValueRange ivs) {
                          SmallVector<Value> lds;
                          for (auto v : ins) {
                            lds.push_back(b.create<AffineLoadOp>(loc, v, ivs));
                          }
                          // assuming ins[0] is LHS and ins[1] is RHS
                          auto op = b.create<arith::CmpFOp>(loc, p, lds[0], lds[1]);
                          b.create<AffineStoreOp>(loc, op, res, ivs);
                        });
    rewriter.replaceOp(op, res.getResult());
    return success();
  }
};

struct CmpIConverter : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently only seeing cmp in the mask calculation, which we're throwing a
    // way right now eventually we'll need to lower these for real, but for now
    // delete them.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReductionConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::triton::populateTritonToAffineConversionPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<SplatConverter, BroadcastConverter, MakeRangeConverter,
           ExpandDimsConverter, AddPtrConverter,
           LoadConverter, StoreConverter,
           LoopConverter, YieldConverter,
           DotConverter,
           ArithConverter<arith::AddFOp>, ArithConverter<arith::AddIOp>,
           ArithConverter<arith::MulIOp>, CmpIConverter,
           ArithConverter<arith::SubFOp>, ArithConverter<arith::DivFOp>,
           ArithConverter<arith::MulFOp>, ArithConverter<arith::SelectOp>,
           NumericCastConverter<arith::TruncFOp>, 
           NumericCastConverter<arith::SIToFPOp>,
           ArithConverter<math::ExpOp>, 
           CmpFConverter<arith::CmpFPredicate::OEQ>,
           ReductionConverter>(patterns.getContext());
}
