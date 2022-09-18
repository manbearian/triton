//===- TritonToAffinePass.cpp - Lowering Triton to standard dialects-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Triton operations to standard dialects.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "triton/Conversion/TritonToAffine/TritonToAffine.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Nepal/IR/NepalUtilities.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace triton;

namespace {
struct TritonToAffinePass : public TritonToAffineBase<TritonToAffinePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, AffineDialect, arith::ArithmeticDialect,
                    math::MathDialect, memref::MemRefDialect, LLVM::LLVMDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                           LLVM::LLVMDialect, math::MathDialect,
                           scf::SCFDialect>();

    target.addIllegalDialect<triton::TritonDialect>();

    // NYI Triton ops 
    target.addLegalOp<triton::GetProgramIdOp>(); // convert to parameter?

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Operations with tensors need to be rewritten to use memrefs
    target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
      return llvm::all_of(op->getResultTypes(),
                         [](Type t) { return !t.isa<TensorType>(); });
    });
    target.addDynamicallyLegalDialect<arith::ArithmeticDialect,
                                      math::MathDialect>(&arithIsLegal);

    auto func = getOperation();
    triton::populateTritonToAffineConversionPatterns(patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), func.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }

private:

  static bool arithIsLegal(Operation *op) {
    // TODO TODO TODO
    // WHAT TO DO HERE -- arithemtic op on pointers end up of tensors of pointers
    if (isa<arith::ConstantOp>(op))
      return true;
    return llvm::all_of(op->getResultTypes(), [](Type t) { return !t.isa<TensorType>(); });
  }

};
} // namespace

std::unique_ptr<Pass> triton::createTritonToAffinePass() {
  return std::make_unique<TritonToAffinePass>();
}
