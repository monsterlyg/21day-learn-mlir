module {
  func.func @fold_add_zero(%x: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    %y = arith.addf %x, %c0 : f32
    return %y : f32
  }
}
