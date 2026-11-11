module {
  func.func @nested_relu(%x: tensor<4xf32>) -> tensor<4xf32> {
    %a = mynpu.relu %x, 0 : tensor<4xf32> -> tensor<4xf32>
    %b = mynpu.relu %a, 0 : tensor<4xf32> -> tensor<4xf32>
    return %b : tensor<4xf32>
  }
}