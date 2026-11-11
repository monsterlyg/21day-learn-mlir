module {
  func.func @main(%arg0: f32) -> f32 {
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %a = arith.addf %arg0, %c1 : f32
    %b = arith.mulf %a, %c2 : f32
    return %b : f32
  }
}