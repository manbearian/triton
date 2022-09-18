//===- TritonToAffine.h - Convert Triton to standard dialects -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes to convert Triton dialect to standard dialects 
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOAFFINE_TRITONTOAFFINE_H
#define TRITON_CONVERSION_TRITONTOAFFINE_TRITONTOAFFINE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createTritonToAffinePass();

/// Populates passes to convert from Triton dialect to Vector dialect
void addTritonToAffinePasses(OpPassManager &pm);

/// Populates conversion passes from Triton dialect to Vector dialect.
void populateTritonToAffineConversionPatterns(RewritePatternSet &patterns);


} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOAFFINE_TRITONTOAFFINE_H
