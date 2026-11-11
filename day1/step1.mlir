module {
    func.func @main(%arg0: f32) -> f32 {
        %c1 = arith.constant 1.0 : f32
        %re = arith.addf %arg0, %c1 : f32
        return %re : f32
    }
}