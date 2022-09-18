add_mlir_dialect_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithmeticDialect
  MLIRSCFDialect
  MLIRFuncDialect
)
