module {
  func.func @conv_relu_test(
      %input: tensor<1x3x224x224xf32>,
      %weight: tensor<16x3x3x3xf32>,
      %bias: tensor<16xf32>) -> tensor<1x16x222x222xf32> {

    %0 = "mynpu.conv"(%input, %weight, %bias) {
      stride = 1 : i64,
      padding = 0 : i64,
      quant = #mynpu.quant<1, 0>
    } : (
      tensor<1x3x224x224xf32>,
      tensor<16x3x3x3xf32>,
      tensor<16xf32>
    ) -> tensor<1x16x222x222xf32>

    %1 = "mynpu.relu"(%0) {axis = 1 : i64}
      : (tensor<1x16x222x222xf32>) -> tensor<1x16x222x222xf32>

    func.return %1 : tensor<1x16x222x222xf32>
  }
}