mlir-opt \
  --load-dialect-plugin=$PWD/build/libMyNPU.so \
  --load-pass-plugin=$PWD/build/libMyNPU.so \
  --pass-pipeline='builtin.module(lower-mynpu-to-arith)' \
  test_lowering.mlir