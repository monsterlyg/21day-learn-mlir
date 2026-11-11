#!/usr/bin/env bash
set -euo pipefail

MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
PLUGIN="$(dirname "$0")/../build/libFusionPlugin.so"
DIR="$(cd "$(dirname "$0")" && pwd)"

# 注意：插件 pass 不能写成顶层 `--pass-name`（cl 在解析命令行前就固定了
# pass 名字表，插件是在解析过程中才被加载注册的），必须用 `-p` 文本流水线
# 并且用锚点 op 包裹（我们的 pass 作用在 builtin.module 上）。
echo "============================================================"
echo "[1] fold-add-zero : arith.addf(x, 0.0) -> x"
echo "============================================================"
"$MLIR_OPT" --load-pass-plugin "$PLUGIN" \
    -p "builtin.module(fold-add-zero)" "$DIR/add_zero.mlir"

echo
echo "============================================================"
echo "[2] fuse-conv-add-relu : conv + add + relu -> fusion.conv_add_relu"
echo "============================================================"
echo "--- before ---"
"$MLIR_OPT" "$DIR/resnet_block.mlir"
echo "--- after ---"
"$MLIR_OPT" --load-dialect-plugin "$PLUGIN" \
    -p "builtin.module(fuse-conv-add-relu)" "$DIR/resnet_block.mlir"

echo
echo "============================================================"
echo "[3] fuse-conv-bn-relu : conv + bn(mul+add) + relu -> fusion.conv_bn_relu"
echo "============================================================"
echo "--- before ---"
"$MLIR_OPT" "$DIR/resnet_block_bn.mlir"
echo "--- after ---"
"$MLIR_OPT" --load-dialect-plugin "$PLUGIN" \
    -p "builtin.module(fuse-conv-bn-relu)" "$DIR/resnet_block_bn.mlir"
