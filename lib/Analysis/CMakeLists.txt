add_mlir_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)