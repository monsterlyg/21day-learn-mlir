#ifndef MYNPU_ATTRS
#define MYNPU_ATTRS

def MyNPU_QuantAttr : AttrDef<MyNPU_Dialect, "quant"> {
    let summary = "Quantization parameters for MyNPU dialect.";
    let mnemonic = "quant";
    let cppNamespace = "::mynpu";
    let cppClassName = "QuantAttr";
    let parameters = (
        ins
        "int64_t":$scale,
        "int64_t":$zeroPoint
    );
    // 加这一行，让它支持文本格式打印和解析
    let assemblyFormat = "`<` $scale `,` $zeroPoint `>`";
}

#endif // MYNPU_ATTRS