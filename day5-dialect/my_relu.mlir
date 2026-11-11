module {
  func.func @relu_test(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    %0 = "mynpu.relu"(%arg0) {axis = 1 : i64} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    func.return %0 : tensor<1x3x224x224xf32>
  }
}
