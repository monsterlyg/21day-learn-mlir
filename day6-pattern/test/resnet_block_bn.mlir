// ResNet block（NHWC）：conv(3x3) -> BN(mul scale, add shift) -> relu
// BN 简化为同 shape 全张量 scale/shift（真实 BN 是逐通道 64xf32 + broadcast，
// 见 README 的「扩展」一节）
func.func @resnet_block_bn(
    %input: tensor<1x56x56x64xf32>,
    %weight: tensor<3x3x64x64xf32>,
    %scale: tensor<1x54x54x64xf32>,
    %shift: tensor<1x54x54x64xf32>,
    %init: tensor<1x54x54x64xf32>) -> tensor<1x54x54x64xf32> {

  %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %weight : tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>)
      outs(%init : tensor<1x54x54x64xf32>) -> tensor<1x54x54x64xf32>

  %mul = linalg.mul
      ins(%conv, %scale : tensor<1x54x54x64xf32>, tensor<1x54x54x64xf32>)
      outs(%init : tensor<1x54x54x64xf32>) -> tensor<1x54x54x64xf32>

  %add = linalg.add
      ins(%mul, %shift : tensor<1x54x54x64xf32>, tensor<1x54x54x64xf32>)
      outs(%init : tensor<1x54x54x64xf32>) -> tensor<1x54x54x64xf32>

  %c0 = arith.constant 0.0 : f32
  %relu = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%add : tensor<1x54x54x64xf32>)
    outs(%init : tensor<1x54x54x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.maximumf %in, %c0 : f32
    linalg.yield %m : f32
  } -> tensor<1x54x54x64xf32>

  return %relu : tensor<1x54x54x64xf32>
}
