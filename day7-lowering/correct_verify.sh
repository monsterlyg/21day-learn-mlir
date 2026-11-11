#!/bin/bash

# MyNPU Lowering Pass 验证脚本
# 用法: ./correct_verify.sh

LLVM_INSTALL_DIR="/data8/liyinggang/Projects/CXX/llvm-project/lyg-install"
BUILD_DIR="/data8/liyinggang/Projects/CXX/mlir-learn/day7-lowering/build"
TEST_FILE="/data8/liyinggang/Projects/CXX/mlir-learn/day7-lowering/test_lowering.mlir"

echo "=========================================="
echo "MyNPU Lowering Pass 验证"
echo "=========================================="

echo ""
echo "✅ 正确命令（关键：两个插件都要加载）:"
echo "  --load-dialect-plugin (注册方言，让解析器认识 mynpu)"
echo "  --load-pass-plugin    (注册 Pass)"
echo ""
echo "运行:"
echo "  mlir-opt \\"
echo "    --load-dialect-plugin=build/libMyNPU.so \\"
echo "    --load-pass-plugin=build/libMyNPU.so \\"
echo "    --pass-pipeline='builtin.module(lower-mynpu-to-arith)' \\"
echo "    test_lowering.mlir"
echo "----------------------------------------"

# 运行 Pass
${LLVM_INSTALL_DIR}/bin/mlir-opt \
    --load-dialect-plugin=${BUILD_DIR}/libMyNPU.so \
    --load-pass-plugin=${BUILD_DIR}/libMyNPU.so \
    --pass-pipeline='builtin.module(lower-mynpu-to-arith)' \
    ${TEST_FILE}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 转换成功！mynpu.addf -> arith.addf"
else
    echo ""
    echo "❌ 转换失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "提示: 只加 --load-pass-plugin 会报错，因为"
echo "解析器不认识 mynpu 方言，必须同时加载"
echo "--load-dialect-plugin"
echo "=========================================="