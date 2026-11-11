module {
  func.func @simple_add_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
  func.func @chain_add_test(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<2x2xf32>
    %1 = arith.addf %0, %arg2 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
  func.func @mixed_ops_test(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4x4xf32>
    %0 = arith.addf %arg0, %cst : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
