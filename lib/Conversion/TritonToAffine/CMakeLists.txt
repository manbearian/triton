add_mlir_conversion_library(TritonToAffine
  TritonToAffine.cpp
  TritonToAffinePass.cpp

  DEPENDS
  TritonConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  )
