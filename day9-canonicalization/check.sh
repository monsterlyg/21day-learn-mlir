MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
PLUGIN=$PWD/build/libMyNPU.so

$MLIR_OPT \
  --load-dialect-plugin=$PLUGIN \
  -p 'builtin.module(canonicalize)' \
  test/nested_relu.mlir

$MLIR_OPT \
  --load-dialect-plugin=$PLUGIN \
  --load-pass-plugin=$PLUGIN \
  -p 'builtin.module(mynpu-cost-analysis)' \
  test/conv.mlir


$MLIR_OPT --load-dialect-plugin=$PLUGIN --load-pass-plugin=$PLUGIN \
  -p 'builtin.module(canonicalize,mynpu-cost-analysis)' test/nested_relu.mlir