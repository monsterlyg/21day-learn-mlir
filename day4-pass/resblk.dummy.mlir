module {
    func.func @resnet_block(%x:f32){
        %c    = arith.constant 1.0:f32
        %conv = arith.mulf %x,%c:f32
        %add  = arith.addf %conv,%x:f32
        %relu = arith.maximumf %add,%c:f32
        return
    }
}