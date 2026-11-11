module {
  func.func @conv(%input: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    %weight = arith.constant dense<0.0> : tensor<64x3x7x7xf32>
    %bias = arith.constant dense<0.0> : tensor<64xf32>
    %output = mynpu.conv %input, %weight, %bias, [2, 2], [3, 3]
      : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32> -> tensor<1x64x112x112xf32>
    return %output : tensor<1x64x112x112xf32>
  }
}