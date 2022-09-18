set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

add_mlir_dialect_library(TritonTransforms
  Combine.cpp

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen
)

target_include_directories(TritonTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
