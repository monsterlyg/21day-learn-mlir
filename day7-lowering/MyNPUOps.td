#ifndef MYNPU_OPS
#define MYNPU_OPS

def MyNPU_ReluOp : Op<MyNPU_Dialect, "relu"> {
    let summary = "Relu operation for MyNPU dialect.";
    let description = [{
        The `mynpu.relu` operation applies the ReLU activation function to its input.

        Example:

        ```mlir
        %0 = mynpu.relu %input : tensor<4x4xf32>
        ```
    }];

    let arguments = (ins AnyTensor:$input, I64Attr:$axis);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = [{
        $input `,` $axis attr-dict `:` type($input) `->` type($output)
    }];
}

def MyNPU_ConvOp : Op<MyNPU_Dialect, "conv"> {
    let arguments = (
        // 1. 操作数 (Operands)
        ins
        AnyTensor:$input,
        AnyTensor:$weight,
        AnyTensor:$bias,

        // 2. 属性 (Attributes) - 直接与操作数混排
        I64Attr:$stride,
        I64Attr:$padding
    );

    let results = (
        outs
        AnyTensor:$output
    );

    let assemblyFormat = [{
        $input `,` $weight `,` $bias `,` $stride `,` $padding attr-dict
        `:` type($input) `,` type($weight) `,` type($bias) `->` type($output)
    }];
}

def MyNPU_AddfOp : Op<MyNPU_Dialect, "addf"> {
    let arguments = (
        // 1. 操作数 (Operands)
        ins
        AnyTensor:$input1,
        AnyTensor:$input2
    );

    let results = (
        outs
        AnyTensor:$output
    );

    let assemblyFormat = [{
        $input1 `,` $input2 attr-dict
        `:` type($input1) `,` type($input2) `->` type($output)
    }];
}

#endif // MYNPU_OPS