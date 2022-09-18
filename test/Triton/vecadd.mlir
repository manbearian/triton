// RUN: triton-opt %s -verify-diagnostics

module {
  func.func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.broadcast %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.broadcast %arg3 : (i32) -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.broadcast %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>
    %9 = tt.broadcast %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %10 = tt.addptr %9, %4 : tensor<256x!tt.ptr<f32>>
    %cst = arith.constant 0.000000e+00 : f32
    %11 = tt.broadcast %cst : (f32) -> tensor<256xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %12 = arith.index_cast %c0_i32 : i32 to index
    %13 = arith.index_cast %arg4 : i32 to index
    %14 = arith.index_cast %c32_i32 : i32 to index
    %15:3 = scf.for %arg6 = %12 to %13 step %14 iter_args(%arg7 = %11, %arg8 = %8, %arg9 = %10) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %18 = tt.broadcast %cst_0 : (f32) -> tensor<256xf32>
      %19 = tt.load %arg8, %6, %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %cst_1 = arith.constant 0.000000e+00 : f32
      %20 = tt.broadcast %cst_1 : (f32) -> tensor<256xf32>
      %21 = tt.load %arg9, %6, %20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %22 = arith.addf %19, %21 : tensor<256xf32>
      %23 = arith.addf %arg7, %22 : tensor<256xf32>
      %24 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %25 = tt.addptr %arg8, %24 : tensor<256x!tt.ptr<f32>>
      %26 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %27 = tt.addptr %arg9, %26 : tensor<256x!tt.ptr<f32>>
      scf.yield %23, %25, %27 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
    }
    %16 = tt.broadcast %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %17 = tt.addptr %16, %4 : tensor<256x!tt.ptr<f32>>
    tt.store %17, %15#0, %6 : tensor<256xf32>
    return
  }
}
// module {
//   func.func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
//     %c64 = arith.constant 64 : index
//     %c32 = arith.constant 32 : index
//     %c0 = arith.constant 0 : index
//     %cst = arith.constant 0.000000e+00 : f32
//     %c256_i32 = arith.constant 256 : i32
//     %0 = tt.get_program_id {axis = 0 : i32} : i32
//     %1 = arith.muli %0, %c256_i32 : i32
//     %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %3 = tt.broadcast %1 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %4 = arith.addi %3, %2 : tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %5 = tt.broadcast %arg3 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %6 = "triton_gpu.cmpi"(%4, %5) {predicate = 2 : i64} : (tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %7 = tt.broadcast %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %8 = tt.addptr %7, %4, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %9 = tt.broadcast %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %10 = tt.addptr %9, %4, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %11 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %12 = arith.index_cast %arg4 : i32 to index
//     %13 = arith.cmpi slt, %c0, %12 : index
//     %14 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %15 = tt.broadcast %13 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %16 = arith.andi %6, %15 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %17 = triton_gpu.copy_async %8, %16, %14 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %18 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %19 = tt.broadcast %13 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %20 = arith.andi %6, %19 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %21 = triton_gpu.copy_async %10, %20, %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %22 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %23 = tt.addptr %8, %22, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %24 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %25 = tt.addptr %10, %24, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %26 = arith.cmpi slt, %c32, %12 : index
//     %27 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %28 = tt.broadcast %26 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %29 = arith.andi %6, %28 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %30 = triton_gpu.copy_async %23, %29, %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %31 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %32 = tt.broadcast %26 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %33 = arith.andi %6, %32 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %34 = triton_gpu.copy_async %25, %33, %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %35 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %36 = tt.addptr %23, %35, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %37 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %38 = tt.addptr %25, %37, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %39 = arith.cmpi slt, %c64, %12 : index
//     %40 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %41 = tt.broadcast %39 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %42 = arith.andi %6, %41 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %43 = triton_gpu.copy_async %36, %42, %40 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %44 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %45 = tt.broadcast %39 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %46 = arith.andi %6, %45 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %47 = triton_gpu.copy_async %38, %46, %44 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %48 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %49 = tt.addptr %36, %48, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %50 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %51 = tt.addptr %38, %50, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %52:12 = scf.for %arg6 = %c0 to %12 step %c32 iter_args(%arg7 = %11, %arg8 = %8, %arg9 = %10, %arg10 = %17, %arg11 = %30, %arg12 = %43, %arg13 = %21, %arg14 = %34, %arg15 = %47, %arg16 = %51, %arg17 = %49, %arg18 = %c64) -> (tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, index) {
//       %55 = arith.addf %arg10, %arg13 : tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %56 = arith.addf %arg7, %55 : tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %57 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %58 = tt.addptr %arg8, %57, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %59 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %60 = tt.addptr %arg9, %59, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %61 = arith.addi %arg18, %c32 : index
//       %62 = arith.cmpi slt, %61, %12 : index
//       %63 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %64 = tt.broadcast %62 : (i1) -> tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %65 = arith.andi %64, %6 : tensor<256xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %66 = triton_gpu.copy_async %arg17, %65, %63 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %67 = tt.broadcast %cst : (f32) -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %68 = triton_gpu.copy_async %arg16, %65, %67 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">> -> tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %69 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %70 = tt.addptr %arg17, %69, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %71 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       %72 = tt.addptr %arg16, %71, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//       scf.yield %56, %58, %60, %arg11, %arg12, %66, %arg14, %arg15, %68, %72, %70, %61 : tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>, index
//     }
//     %53 = tt.broadcast %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     %54 = tt.addptr %53, %4, : tensor<256x!tt.ptr<f32>, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     tt.store %54, %52#0, %6 : tensor<256xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
//     return
//   }
// }
