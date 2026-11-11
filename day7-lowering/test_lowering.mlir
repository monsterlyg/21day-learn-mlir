// 测试文件：验证 MyNPU Lowering Pass
// 这个文件包含 mynpu.addf 操作，运行 lower-mynpu-to-arith pass 后应该转换为 arith.addf

module {
  // 简单的加法测试
  func.func @simple_add_test(
      %input1: tensor<4x4xf32>,
      %input2: tensor<4x4xf32>) -> tensor<4x4xf32> {

    %0 = "mynpu.addf"(%input1, %input2)
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

    func.return %0 : tensor<4x4xf32>
  }

  // 多个加法链式调用测试
  func.func @chain_add_test(
      %a: tensor<2x2xf32>,
      %b: tensor<2x2xf32>,
      %c: tensor<2x2xf32>) -> tensor<2x2xf32> {

    %0 = "mynpu.addf"(%a, %b)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

    %1 = "mynpu.addf"(%0, %c)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

    func.return %1 : tensor<2x2xf32>
  }

  // 与其他操作混合的测试
  func.func @mixed_ops_test(
      %input: tensor<4x4xf32>) -> tensor<4x4xf32> {

    // 创建常量用于加法
    %0 = arith.constant dense<1.0> : tensor<4x4xf32>

    // 使用 mynpu.addf
    %1 = "mynpu.addf"(%input, %0)
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

    func.return %1 : tensor<4x4xf32>
  }
}